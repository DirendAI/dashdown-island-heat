"""Offline unit tests for heat_island.boundary (no network, no osmnx/Nominatim calls).

Only the pure logic is exercised here: `_parse_place` (country/city_id derivation) and the
GeoJSON boundary cache round-trip, using the same helpers `get_city_boundary` itself calls.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import box

from heat_island.boundary import (
    CityBoundary,
    _boundary_cache_path,
    _parse_place,
    _read_boundary_cache,
    _read_query_index,
    _write_boundary_cache,
)
from heat_island.config import PipelineConfig


def test_parse_place_uses_given_name():
    name, country, city_id = _parse_place(
        "Gent", "Gent, Oost-Vlaanderen, Vlaanderen, België / Belgique / Belgien"
    )
    assert name == "Gent"
    assert country == "België / Belgique / Belgien"
    assert city_id == "gent-belgie"


@pytest.mark.parametrize("missing_name", [None, float("nan"), "", "   "])
def test_parse_place_falls_back_to_display_name_when_name_missing(missing_name):
    name, country, city_id = _parse_place(
        missing_name, "Springfield, Sangamon County, Illinois, United States"
    )
    assert name == "Springfield"
    assert country == "United States"
    assert city_id == "springfield-united-states"


def test_parse_place_country_is_first_slash_token_of_last_comma_component():
    name, country, city_id = _parse_place(
        "Brussels", "Brussels, Brussels-Capital, Belgique / België / Belgien"
    )
    assert name == "Brussels"
    # country kept as-is (not just the first token)
    assert country == "Belgique / België / Belgien"
    # city_id derived from the *first* slash-token, ascii-slugified
    assert city_id == "brussels-belgique"


def test_parse_place_no_slash_in_country():
    name, country, city_id = _parse_place("Tokyo", "Tokyo, Japan")
    assert name == "Tokyo"
    assert country == "Japan"
    assert city_id == "tokyo-japan"


def _make_boundary(**overrides) -> CityBoundary:
    geom = box(3.6, 51.0, 3.8, 51.15)
    defaults = dict(
        query="Ghent, Belgium",
        name="Gent",
        country="België / Belgique / Belgien",
        city_id="gent-belgie",
        geometry=geom,
        centroid_lat=geom.centroid.y,
        centroid_lon=geom.centroid.x,
    )
    defaults.update(overrides)
    return CityBoundary(**defaults)


def test_boundary_cache_roundtrip(tmp_path):
    cfg = PipelineConfig(data_dir=tmp_path / "data")
    boundary = _make_boundary()

    path = _write_boundary_cache(cfg, boundary)

    assert path.exists()
    assert path == _boundary_cache_path(cfg, boundary.city_id)
    assert path == cfg.cache_dir / boundary.city_id / "boundary.geojson"

    loaded = _read_boundary_cache(cfg, boundary.city_id)

    assert loaded is not None
    assert loaded.query == boundary.query
    assert loaded.name == boundary.name
    assert loaded.country == boundary.country
    assert loaded.city_id == boundary.city_id
    assert loaded.geometry.equals(boundary.geometry)
    assert math.isclose(loaded.centroid_lat, boundary.centroid_lat)
    assert math.isclose(loaded.centroid_lon, boundary.centroid_lon)


def test_boundary_cache_miss_returns_none(tmp_path):
    cfg = PipelineConfig(data_dir=tmp_path / "data")
    assert _read_boundary_cache(cfg, "does-not-exist") is None


def test_boundary_cache_survives_multipolygon(tmp_path):
    from shapely.geometry import MultiPolygon

    cfg = PipelineConfig(data_dir=tmp_path / "data")
    geom = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    boundary = _make_boundary(city_id="multi-city", geometry=geom, centroid_lat=1.5, centroid_lon=1.5)

    _write_boundary_cache(cfg, boundary)
    loaded = _read_boundary_cache(cfg, boundary.city_id)

    assert loaded.geometry.equals(geom)
    assert loaded.geometry.geom_type == "MultiPolygon"


def test_query_index_written_is_readable_directly(tmp_path):
    """Exercises the same on-disk format get_city_boundary's cache-hit path reads."""
    from heat_island.boundary import _write_query_index_entry

    cfg = PipelineConfig(data_dir=tmp_path / "data")
    _write_query_index_entry(cfg, "ghent-belgium", "gent-belgie")

    index = _read_query_index(cfg)
    assert index == {"ghent-belgium": "gent-belgie"}
