"""H3 (v4 API) hex grid helpers: polygon -> cells, cells -> GeoDataFrame, point aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .util import get_logger, utm_crs_for

log = get_logger(__name__)


def polygon_to_hexes(geom: BaseGeometry, res: int) -> list[str]:
    """All H3 cells (at `res`) whose center falls inside `geom` (EPSG:4326).

    Handles MultiPolygon by iterating parts and unioning cell sets. If the polygon is too
    small to contain any cell center, falls back to the cell containing the centroid plus its
    1-ring neighbors, so this never returns an empty list.
    """
    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]

    cells: set[str] = set()
    for part in parts:
        shape_ = h3.geo_to_h3shape(part.__geo_interface__)
        cells |= set(h3.h3shape_to_cells(shape_, res))

    if not cells:
        centroid = geom.centroid
        origin = h3.latlng_to_cell(centroid.y, centroid.x, res)
        cells |= set(h3.grid_disk(origin, 1))
        log.warning(
            "polygon_to_hexes: no cell centers inside geometry at res %d — "
            "falling back to %d cells around the centroid",
            res,
            len(cells),
        )

    result = sorted(cells)
    log.info("polygon_to_hexes: %d cells at res %d", len(result), res)
    return result


def hexes_to_gdf(cells: list[str]) -> gpd.GeoDataFrame:
    """Cells -> GeoDataFrame with columns h3, lat, lon, geometry (EPSG:4326, closed rings)."""
    rows = []
    for cell in cells:
        lat, lon = h3.cell_to_latlng(cell)
        boundary = h3.cell_to_boundary(cell)  # tuple of (lat, lng) pairs
        ring = [(lng, lat) for lat, lng in boundary]  # shapely wants (lon, lat)
        ring.append(ring[0])  # close the ring explicitly
        rows.append({"h3": cell, "lat": lat, "lon": lon, "geometry": Polygon(ring)})

    return gpd.GeoDataFrame(rows, columns=["h3", "lat", "lon", "geometry"], crs="EPSG:4326")


def points_to_hex_means(
    df: pd.DataFrame,
    res: int,
    value_cols: list[str],
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Assign each row to an H3 cell (at `res`) and average `value_cols` per cell.

    Rows with NaN lat/lon are dropped first (can't be assigned a cell). Averaging itself uses
    pandas' default skipna groupby-mean, so a NaN in one value column doesn't drop the row for
    other columns.
    """
    valid = df.dropna(subset=[lat_col, lon_col])
    cells = [h3.latlng_to_cell(lat, lon, res) for lat, lon in zip(valid[lat_col], valid[lon_col])]
    valid = valid.assign(h3=cells)
    return valid.groupby("h3")[value_cols].mean().reset_index()


def hex_parents(cells: Iterable[str], offset: int) -> np.ndarray:
    """Parent cell of each cell at resolution `max(res(c) - offset, 0)` (used for spatial CV groups)."""
    return np.array([h3.cell_to_parent(c, max(h3.get_resolution(c) - offset, 0)) for c in cells])


def grid_stats(cells: Iterable[str], geom: BaseGeometry) -> dict:
    """n_hexes, boundary area (UTM, km²), and mean per-hex area (km²)."""
    cell_list = list(cells)
    n_hexes = len(cell_list)

    boundary_gs = gpd.GeoSeries([geom], crs="EPSG:4326")
    utm_crs = utm_crs_for(boundary_gs)
    area_km2 = float(boundary_gs.to_crs(utm_crs).area.sum() / 1e6)

    if n_hexes:
        hex_gdf = hexes_to_gdf(cell_list)
        total_hex_area_km2 = float(hex_gdf.to_crs(utm_crs).area.sum() / 1e6)
        mean_hex_area_km2 = total_hex_area_km2 / n_hexes
    else:
        mean_hex_area_km2 = 0.0

    return {"n_hexes": n_hexes, "area_km2": area_km2, "mean_hex_area_km2": mean_hex_area_km2}
