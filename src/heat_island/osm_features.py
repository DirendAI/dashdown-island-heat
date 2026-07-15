"""OSM-derived per-hex features: building/road density, distance to water/parks, park flag.

Buildings and roads are fetched on the exact city boundary; water and parks are fetched on the
boundary buffered by `cfg.buffer_m` so a canal or park just outside the administrative line
still pulls `dist_*` down for hexes near the edge. Every metric quantity (areas, lengths,
distances) is computed in the city's local UTM CRS — never in EPSG:4326 — per ARCHITECTURE.md's
global conventions.

The geometry math is split into small, private, network-free helpers (`_building_density`,
`_road_density`, `_distance_to_nearest`, `_is_park`) so it can be unit-tested with synthetic
GeoDataFrames offline; `fetch_osm_features_per_hex` is the only function that touches the
network (via osmnx/Overpass), and it is the only thing tests must not call.
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from .boundary import CityBoundary, _configure_osmnx
from .config import PipelineConfig
from .util import get_logger, retry_call, utm_crs_for

log = get_logger(__name__)

_POLY_TYPES = ("Polygon", "MultiPolygon")
_WATER_TAGS: dict = {"natural": "water", "water": True, "waterway": ["riverbank"]}
_PARK_TAGS: dict = {
    "leisure": ["park", "garden", "nature_reserve"],
    "landuse": ["recreation_ground", "village_green", "cemetery"],
}


# --------------------------------------------------------------------------------------
# Pure geometry helpers (offline-testable) — everything here takes/returns UTM geometries.
# --------------------------------------------------------------------------------------


def _building_density(
    buildings_utm: gpd.GeoDataFrame,
    hex_utm_by_h3: gpd.GeoDataFrame,
    res: int,
) -> pd.Series:
    """Sum of building footprint area (UTM m²) per hex, divided by hex area, clipped [0, 1].

    Buildings are assigned to the hex containing their *centroid*: centroids are computed in
    UTM (vectorized `.centroid` over the whole GeoSeries — no per-row `.apply`), reprojected
    back to EPSG:4326 in a single batched `to_crs` call, then matched to an H3 cell with
    `h3.latlng_to_cell` via a list comprehension (h3 has no vectorized API; a comprehension is
    the fast path per hexgrid.py's own convention). Hexes with no buildings get 0. A building
    whose centroid falls outside every hex in `hex_utm_by_h3` (possible right at the boundary)
    is silently dropped — it cannot contribute to any hex in the grid.

    Returns a Series of density values indexed by `h3`, one entry per row of `hex_utm_by_h3`
    (i.e. aligned with the full hex grid, not just hexes that got a building).
    """
    hex_area = hex_utm_by_h3.geometry.area
    if buildings_utm is None or len(buildings_utm) == 0:
        return pd.Series(0.0, index=hex_area.index)

    centroids_4326 = buildings_utm.geometry.centroid.to_crs("EPSG:4326")
    cells = [h3.latlng_to_cell(pt.y, pt.x, res) for pt in centroids_4326]
    areas_m2 = buildings_utm.geometry.area.to_numpy()

    per_hex = pd.Series(areas_m2, index=cells).groupby(level=0).sum()
    per_hex = per_hex.reindex(hex_area.index, fill_value=0.0)

    density = (per_hex / hex_area).clip(lower=0.0, upper=1.0)
    return density


def _road_density(
    edges_utm: gpd.GeoDataFrame,
    hex_utm_by_h3: gpd.GeoDataFrame,
    res: int,
) -> pd.Series:
    """Total road length (km) per hex, divided by hex area (km²) -> km of road per km².

    Each edge is assigned to the hex containing its geometry *midpoint*
    (`.interpolate(0.5, normalized=True)` computed in UTM, then a single batched reprojection
    to EPSG:4326). Edge length comes from the graph's own `length` column (meters, computed by
    osmnx independently of any later reprojection) when present, else the UTM geometry length.

    Returns a Series indexed by `h3`, aligned with the full hex grid (0 where no roads).
    """
    hex_area_km2 = hex_utm_by_h3.geometry.area / 1e6
    if edges_utm is None or len(edges_utm) == 0:
        return pd.Series(0.0, index=hex_area_km2.index)

    if "length" in edges_utm.columns:
        length_m = edges_utm["length"].astype(float).to_numpy()
    else:
        length_m = edges_utm.geometry.length.to_numpy()

    midpoints_utm = edges_utm.geometry.interpolate(0.5, normalized=True)
    midpoints_4326 = gpd.GeoSeries(midpoints_utm, crs=edges_utm.crs).to_crs("EPSG:4326")
    cells = [h3.latlng_to_cell(pt.y, pt.x, res) for pt in midpoints_4326]

    per_hex_m = pd.Series(length_m, index=cells).groupby(level=0).sum()
    per_hex_m = per_hex_m.reindex(hex_area_km2.index, fill_value=0.0)

    return (per_hex_m / 1000.0) / hex_area_km2


def _distance_to_nearest(hex_centroids_utm, polys_utm, cap: float) -> np.ndarray:
    """UTM-meter distance from each hex centroid to the nearest polygon, capped at `cap`.

    Fully vectorized via a single `shapely.STRtree.query_nearest` call (no polygons found —
    e.g. an empty `polys_utm` because Overpass returned nothing — every hex simply keeps the
    cap). `all_matches=False` guarantees at most one tree match per input point, so the
    returned (input_idx, tree_idx) pairs can be scattered straight into the output array.
    """
    n = len(hex_centroids_utm)
    out = np.full(n, float(cap), dtype=float)
    if n == 0 or polys_utm is None or len(polys_utm) == 0:
        return out

    tree = STRtree(polys_utm)
    idx, dist = tree.query_nearest(hex_centroids_utm, all_matches=False, return_distance=True)
    out[idx[0]] = np.minimum(dist, cap)
    return out


def _is_park(hex_centroids_utm, parks_utm) -> np.ndarray:
    """True where a hex centroid falls strictly within any park polygon.

    Vectorized via `STRtree.query(points, predicate="within")` over a tree built on the park
    polygons: for each (point, tree_geometry) candidate pair whose bounding boxes intersect,
    shapely tests `point.within(park_polygon)`. No parks -> an empty tree -> all False.
    """
    n = len(hex_centroids_utm)
    out = np.zeros(n, dtype=bool)
    if n == 0 or parks_utm is None or len(parks_utm) == 0:
        return out

    tree = STRtree(parks_utm)
    matched_idx = tree.query(hex_centroids_utm, predicate="within")[0]
    out[np.unique(matched_idx)] = True
    return out


# --------------------------------------------------------------------------------------
# Fetch + cache orchestration (network — not exercised by unit tests).
# --------------------------------------------------------------------------------------


def _cache_path(cfg: PipelineConfig, city_id: str, res: int) -> Path:
    return cfg.city_cache(city_id) / f"osm_res{res}.parquet"


def _only_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf[gdf.geometry.geom_type.isin(_POLY_TYPES)]


def _empty_gdf(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], crs=crs)}, geometry="geometry", crs=crs)


def _to_utm_polygons(raw: gpd.GeoDataFrame | None, utm_crs) -> gpd.GeoDataFrame:
    """None/empty/no-polygon-rows -> empty UTM gdf; else polygonal rows projected to UTM."""
    if raw is None or len(raw) == 0:
        return _empty_gdf(utm_crs)
    polys = _only_polygons(raw)
    if len(polys) == 0:
        return _empty_gdf(utm_crs)
    return polys.to_crs(utm_crs)


def _buffered_polygon_4326(boundary: CityBoundary, cfg: PipelineConfig, utm_crs) -> BaseGeometry:
    """Boundary polygon buffered by `cfg.buffer_m`, computed in UTM, returned in EPSG:4326."""
    buffered_utm = gpd.GeoSeries([boundary.geometry], crs="EPSG:4326").to_crs(utm_crs).buffer(cfg.buffer_m)
    return buffered_utm.to_crs("EPSG:4326").iloc[0]


def _safe_fetch(fn, what: str):
    """retry_call(fn); on total failure (Overpass down, zero results, ...) log WARNING, return None."""
    try:
        return retry_call(fn, what=what)
    except Exception as exc:
        log.warning("%s failed after retries (%s) — falling back to empty result", what, exc)
        return None


def _fetch_road_edges(boundary: CityBoundary) -> gpd.GeoDataFrame:
    graph = ox.graph.graph_from_polygon(boundary.geometry, network_type="drive", retain_all=True)
    return ox.convert.graph_to_gdfs(graph, nodes=False)


def fetch_osm_features_per_hex(
    boundary: CityBoundary, hex_gdf: gpd.GeoDataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    """Per-hex OSM features: h3, building_density, road_density, dist_water_m, dist_park_m, is_park.

    Cached at `cache/<city_id>/osm_res{h3_resolution}.parquet` — a cache hit skips every
    network call. Buildings/roads/water/parks each get 4 retried attempts (`util.retry_call`);
    if a fetch still fails (or Overpass genuinely has zero matching features), that layer logs
    a WARNING and degrades gracefully (buildings/roads -> zero density; water/parks -> hexes
    capped at `cfg.dist_cap_m` and `is_park` all False) rather than crashing the pipeline.
    """
    res = cfg.h3_resolution
    city_dir = cfg.city_cache(boundary.city_id)
    cache_path = _cache_path(cfg, boundary.city_id, res)
    if cache_path.exists():
        log.info("%s: OSM features cache hit (%s)", boundary.name, cache_path)
        return pd.read_parquet(cache_path)

    _configure_osmnx(cfg)
    osm_http_cache = city_dir / "osm_http"
    osm_http_cache.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(osm_http_cache)

    utm_crs = utm_crs_for(hex_gdf)
    hex_gdf_utm = hex_gdf.to_crs(utm_crs)
    hex_utm_by_h3 = hex_gdf_utm.set_index("h3")
    hex_centroids_utm = hex_gdf_utm.geometry.centroid

    log.info("%s: fetching OSM buildings (%d hexes)", boundary.name, len(hex_gdf))
    buildings_raw = _safe_fetch(
        lambda: ox.features.features_from_polygon(boundary.geometry, {"building": True}),
        what=f"OSM buildings for {boundary.name}",
    )
    time.sleep(1)

    log.info("%s: fetching OSM road network", boundary.name)
    edges_raw = _safe_fetch(
        lambda: _fetch_road_edges(boundary), what=f"OSM road graph for {boundary.name}"
    )
    time.sleep(1)

    buffered_4326 = _buffered_polygon_4326(boundary, cfg, utm_crs)

    log.info("%s: fetching OSM water (buffer %.0f m)", boundary.name, cfg.buffer_m)
    water_raw = _safe_fetch(
        lambda: ox.features.features_from_polygon(buffered_4326, _WATER_TAGS),
        what=f"OSM water for {boundary.name}",
    )
    time.sleep(1)

    log.info("%s: fetching OSM parks (buffer %.0f m)", boundary.name, cfg.buffer_m)
    parks_raw = _safe_fetch(
        lambda: ox.features.features_from_polygon(buffered_4326, _PARK_TAGS),
        what=f"OSM parks for {boundary.name}",
    )

    buildings_utm = _to_utm_polygons(buildings_raw, utm_crs)
    edges_utm = edges_raw.to_crs(utm_crs) if edges_raw is not None and len(edges_raw) else _empty_gdf(utm_crs)

    water_utm_all = _to_utm_polygons(water_raw, utm_crs)
    water_utm = water_utm_all[water_utm_all.geometry.area >= cfg.min_water_area_m2]

    parks_utm = _to_utm_polygons(parks_raw, utm_crs)

    building_density = _building_density(buildings_utm, hex_utm_by_h3, res)
    road_density = _road_density(edges_utm, hex_utm_by_h3, res)
    dist_water_m = _distance_to_nearest(hex_centroids_utm, water_utm.geometry, cfg.dist_cap_m)
    dist_park_m = _distance_to_nearest(hex_centroids_utm, parks_utm.geometry, cfg.dist_cap_m)
    is_park = _is_park(hex_centroids_utm, parks_utm.geometry)

    out = pd.DataFrame({"h3": hex_gdf["h3"].to_numpy()})
    out["building_density"] = out["h3"].map(building_density).fillna(0.0).to_numpy()
    out["road_density"] = out["h3"].map(road_density).fillna(0.0).to_numpy()
    out["dist_water_m"] = dist_water_m
    out["dist_park_m"] = dist_park_m
    out["is_park"] = is_park

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)

    log.info(
        "%s: OSM features ready — %d buildings, %d road edges, %d water polys (>=%.0fm²), "
        "%d park polys | building_density mean=%.3f road_density mean=%.2f km/km² "
        "dist_water_m median=%.0f is_park sum=%d",
        boundary.name, len(buildings_utm), len(edges_utm), len(water_utm), cfg.min_water_area_m2,
        len(parks_utm), out["building_density"].mean(), out["road_density"].mean(),
        out["dist_water_m"].median(), int(out["is_park"].sum()),
    )
    return out
