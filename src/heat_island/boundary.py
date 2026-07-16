"""Geocode a city query to a cached polygon boundary via Nominatim (osmnx)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

from .config import PipelineConfig
from .util import CityNotFoundError, get_logger, retry_call, slugify

log = get_logger(__name__)


@dataclass
class CityBoundary:
    """A geocoded city: canonical identity + its boundary polygon (EPSG:4326)."""

    query: str  # user's original query string
    name: str  # canonical short name from Nominatim (e.g. "Gent")
    country: str  # last component of display_name, stripped
    city_id: str  # util.slugify(f"{name}-{country_first_token}") — stable across query spellings
    geometry: BaseGeometry  # shapely (Multi)Polygon, EPSG:4326
    centroid_lat: float
    centroid_lon: float


def _configure_osmnx(cfg: PipelineConfig) -> None:
    """Point osmnx's HTTP cache + user agent at our cache dir. Call before any osmnx network call.

    The geocode step doesn't know city_id yet, so it shares one "osm_http" folder under the
    cache root (per-city osmnx caching for features/graphs happens later, in osm_features.py).
    """
    cache_folder = cfg.cache_dir / "osm_http"
    cache_folder.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(cache_folder)
    ox.settings.http_user_agent = "heat-island-mapper/0.1 (github.com/DirendAI/dashdown-island-heat)"
    ox.settings.requests_timeout = 180
    ox.settings.log_console = False


def _parse_place(name: Any, display_name: str) -> tuple[str, str, str]:
    """Pure derivation of (name, country, city_id) from Nominatim's `name` / `display_name`.

    - `name` missing/NaN/blank -> fall back to the first comma-component of `display_name`.
    - `country` = last comma-component of `display_name`, stripped (kept as-is, e.g. a
      multilingual "België / Belgique / Belgien" string).
    - `city_id` = slugify(f"{name}-{first '/'-token of country}").

    No network access — this is the pure, unit-testable half of get_city_boundary.
    """
    if pd.isna(name) or not str(name).strip():
        resolved_name = display_name.split(",")[0].strip()
    else:
        resolved_name = str(name).strip()

    country = display_name.split(",")[-1].strip()
    country_first_token = country.split("/")[0].strip()
    city_id = slugify(f"{resolved_name}-{country_first_token}")
    return resolved_name, country, city_id


def _query_index_path(cfg: PipelineConfig) -> Path:
    return cfg.cache_dir / "query_index.json"


def _boundary_cache_path(cfg: PipelineConfig, city_id: str) -> Path:
    return cfg.cache_dir / city_id / "boundary.geojson"


def _read_query_index(cfg: PipelineConfig) -> dict[str, str]:
    path = _query_index_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_query_index_entry(cfg: PipelineConfig, query_slug: str, city_id: str) -> None:
    path = _query_index_path(cfg)
    index = _read_query_index(cfg)
    index[query_slug] = city_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True))


def _boundary_to_geojson(boundary: CityBoundary) -> dict:
    """Serialize a CityBoundary to a single-feature GeoJSON FeatureCollection."""
    feature = {
        "type": "Feature",
        "geometry": mapping(boundary.geometry),
        "properties": {
            "query": boundary.query,
            "name": boundary.name,
            "country": boundary.country,
            "city_id": boundary.city_id,
        },
    }
    return {"type": "FeatureCollection", "features": [feature]}


def _geojson_to_boundary(data: dict) -> CityBoundary:
    """Reconstruct a CityBoundary from the cached GeoJSON FeatureCollection.

    The centroid isn't stored in the cache; it's cheap and deterministic to recompute from
    the geometry (EPSG:4326 is fine for a display centroid).
    """
    feature = data["features"][0]
    geom = shape(feature["geometry"])
    props = feature["properties"]
    centroid = geom.centroid
    return CityBoundary(
        query=props["query"],
        name=props["name"],
        country=props["country"],
        city_id=props["city_id"],
        geometry=geom,
        centroid_lat=float(centroid.y),
        centroid_lon=float(centroid.x),
    )


def _write_boundary_cache(cfg: PipelineConfig, boundary: CityBoundary) -> Path:
    path = _boundary_cache_path(cfg, boundary.city_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_boundary_to_geojson(boundary)))
    return path


def _read_boundary_cache(cfg: PipelineConfig, city_id: str) -> CityBoundary | None:
    path = _boundary_cache_path(cfg, city_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return _geojson_to_boundary(data)
    except (json.JSONDecodeError, OSError, KeyError, IndexError) as exc:
        log.warning("boundary cache at %s is unreadable (%s) — ignoring", path, exc)
        return None


def get_city_boundary(query: str, cfg: PipelineConfig) -> CityBoundary:
    """Resolve a free-text city query to a cached CityBoundary.

    Cache hit (query seen before, boundary GeoJSON present) skips Nominatim entirely.
    """
    query_slug = slugify(query)
    cached_city_id = _read_query_index(cfg).get(query_slug)
    if cached_city_id:
        cached = _read_boundary_cache(cfg, cached_city_id)
        if cached is not None:
            log.info("boundary cache hit for %r -> city_id=%s", query, cached_city_id)
            return cached

    _configure_osmnx(cfg)
    log.info("geocoding %r via Nominatim", query)
    try:
        gdf = retry_call(ox.geocoder.geocode_to_gdf, query, what=f"geocode {query!r}")
    except Exception as exc:
        reason = str(exc).rstrip(".")
        raise CityNotFoundError(
            f"Could not geocode '{query}': {reason}. Try a more specific query, "
            f"e.g. 'Springfield, Illinois, USA' instead of 'Springfield'."
        ) from exc

    if gdf is None or len(gdf) == 0:
        raise CityNotFoundError(
            f"Nominatim geocoder returned no results for '{query}'. Check the spelling or "
            f"try a more specific query, e.g. 'Springfield, Illinois, USA'."
        )

    row = gdf.iloc[0]
    geom = row.geometry
    if not isinstance(geom, (Polygon, MultiPolygon)):
        geom_type = geom.geom_type if geom is not None else "null geometry"
        raise CityNotFoundError(
            f"'{query}' geocoded to a {geom_type}, not a polygon — try qualifying it, "
            f"e.g. '{query}, Country'."
        )

    display_name = str(row["display_name"])
    raw_name = row["name"] if "name" in gdf.columns else None
    name, country, city_id = _parse_place(raw_name, display_name)

    centroid = geom.centroid
    boundary = CityBoundary(
        query=query,
        name=name,
        country=country,
        city_id=city_id,
        geometry=geom,
        centroid_lat=float(centroid.y),
        centroid_lon=float(centroid.x),
    )

    _write_boundary_cache(cfg, boundary)
    _write_query_index_entry(cfg, query_slug, city_id)
    log.info("resolved %r -> city_id=%s (%s, %s)", query, city_id, name, country)
    return boundary
