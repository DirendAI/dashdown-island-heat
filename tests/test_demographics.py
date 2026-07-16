"""Offline unit tests for heat_island.demographics (no network — `fetch_json` is always faked).

TIGERweb/ACS responses are hand-built: two adjacent rectangular "tracts" in UTM covering a
6-hex cluster, plus a tiny 2-row ACS table (one tract with a sentinel income). Where a test
needs an "expected" area-weighted value, it is computed from the *actual* shapely intersection
of the exact (round-tripped-through-4326) geometries the implementation itself uses — never
hand-derived — so the test can't quietly encode the same arithmetic bug as the implementation.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import box, mapping

from heat_island.boundary import CityBoundary
from heat_island.config import PipelineConfig
from heat_island.demographics import (
    ACS_VARS,
    OVER65_VARS,
    TIGERWEB_MAPSERVER,
    UNDER5_VARS,
    _fetch_tracts,
    _find_tract_layer_id,
    _query_acs,
    fetch_demographics_per_hex,
)
from heat_island.hexgrid import hexes_to_gdf

UTM_CRS = 32631  # correct UTM zone for ~3.7°E
LAT0, LON0 = 51.05, 3.72
RES = 9

GEOID_A = "12345000100"
GEOID_B = "12345000200"

# Two adjacent rectangles in UTM, split at x=551000, sized to cover the 6-hex cluster below:
# hex 0 falls fully inside A, hex 1 straddles the A/B line, hexes 2-5 fall fully inside B.
# (Verified against the actual hex geometries produced by _hex_cluster/hexes_to_gdf below.)
_TRACT_A_UTM = box(500_000.0, 5_650_000.0, 551_000.0, 5_660_000.0)
_TRACT_B_UTM = box(551_000.0, 5_650_000.0, 600_000.0, 5_660_000.0)

INCOME_A = 60_000.0
TOTAL_A = 1000.0
UNDER5_EACH_A = 20.0
OVER65_EACH_A = 10.0
UNDER5_A = UNDER5_EACH_A * len(UNDER5_VARS)
OVER65_A = OVER65_EACH_A * len(OVER65_VARS)
PCT_OVER_65_A = 100.0 * OVER65_A / TOTAL_A  # 12.0
PCT_UNDER_5_A = 100.0 * UNDER5_A / TOTAL_A  # 4.0

INCOME_B_SENTINEL = -666_666_666.0  # Census "not applicable" sentinel
TOTAL_B = 2000.0
UNDER5_EACH_B = 30.0
OVER65_EACH_B = 15.0
UNDER5_B = UNDER5_EACH_B * len(UNDER5_VARS)
OVER65_B = OVER65_EACH_B * len(OVER65_VARS)
PCT_OVER_65_B = 100.0 * OVER65_B / TOTAL_B  # 9.0
PCT_UNDER_5_B = 100.0 * UNDER5_B / TOTAL_B  # 3.0


def _hex_cluster(n: int = 6) -> list[str]:
    """`n` distinct res-9 H3 cells spaced out along a line near (LAT0, LON0)."""
    cells: list[str] = []
    seen: set[str] = set()
    i = 0
    while len(cells) < n:
        cell = h3.latlng_to_cell(LAT0, LON0 + i * 0.006, RES)
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)
        i += 1
    return cells


def _make_boundary(country: str) -> CityBoundary:
    geom = box(3.6, 51.0, 3.8, 51.15)
    centroid = geom.centroid
    return CityBoundary(
        query="Test City, Testland",
        name="Test City",
        country=country,
        city_id="test-city-testland",
        geometry=geom,
        centroid_lat=centroid.y,
        centroid_lon=centroid.x,
    )


US_BOUNDARY = _make_boundary("United States")
NON_US_BOUNDARY = _make_boundary("België / Belgique / Belgien")


@pytest.fixture(scope="module")
def hex_cells() -> list[str]:
    return _hex_cluster(6)


@pytest.fixture(scope="module")
def hex_gdf(hex_cells) -> gpd.GeoDataFrame:
    return hexes_to_gdf(hex_cells)


# --------------------------------------------------------------------------------------
# Synthetic TIGERweb / ACS fixtures
# --------------------------------------------------------------------------------------


def _tract_feature(geoid: str, geom_utm) -> dict:
    """A GeoJSON Feature exactly like TIGERweb's `f=geojson` query would return (EPSG:4326)."""
    geom_4326 = gpd.GeoSeries([geom_utm], crs=UTM_CRS).to_crs("EPSG:4326").iloc[0]
    return {"type": "Feature", "properties": {"GEOID": geoid}, "geometry": mapping(geom_4326)}


def _tracts_geojson() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [_tract_feature(GEOID_A, _TRACT_A_UTM), _tract_feature(GEOID_B, _TRACT_B_UTM)],
        "exceededTransferLimit": False,
    }


LAYER_LIST_JSON = {
    "layers": [
        {"id": 0, "name": "Census Tracts"},
        {"id": 12, "name": "Census Blocks"},
    ]
}


def _acs_header() -> list[str]:
    return [*ACS_VARS, "state", "county", "tract"]


def _acs_row(geoid: str, income: float, total: float, under5_each: float, over65_each: float) -> list[str]:
    state, county, tract = geoid[0:2], geoid[2:5], geoid[5:11]
    under5 = [under5_each] * len(UNDER5_VARS)
    over65 = [over65_each] * len(OVER65_VARS)
    values = [income, total, *under5, *over65]
    return [str(v) for v in values] + [state, county, tract]


ROW_A = _acs_row(GEOID_A, INCOME_A, TOTAL_A, UNDER5_EACH_A, OVER65_EACH_A)
ROW_B = _acs_row(GEOID_B, INCOME_B_SENTINEL, TOTAL_B, UNDER5_EACH_B, OVER65_EACH_B)
ACS_ROWS = [_acs_header(), ROW_A, ROW_B]


def _make_fake_fetch_json(*, acs_rows=None, tracts_geojson=None, layer_list=None):
    """A URL-dispatching fake `fetch_json`. Records every call in `.calls` for inspection."""
    acs_rows = ACS_ROWS if acs_rows is None else acs_rows
    tracts_geojson = _tracts_geojson() if tracts_geojson is None else tracts_geojson
    layer_list = LAYER_LIST_JSON if layer_list is None else layer_list
    calls: list[tuple[str, dict]] = []

    def fetch_json(url: str, params: dict):
        calls.append((url, dict(params)))
        if url == TIGERWEB_MAPSERVER:
            return layer_list
        if url.startswith(TIGERWEB_MAPSERVER + "/"):
            return tracts_geojson
        if url.startswith("https://api.census.gov/"):
            return acs_rows
        raise AssertionError(f"unexpected url in test fake: {url}")

    fetch_json.calls = calls  # type: ignore[attr-defined]
    return fetch_json


# --------------------------------------------------------------------------------------
# Gate + error handling
# --------------------------------------------------------------------------------------


def test_non_us_country_short_circuits_before_any_fetch(tmp_path, hex_gdf):
    calls: list[tuple[str, dict]] = []

    def fetch_json(url, params):
        calls.append((url, params))
        raise AssertionError("fetch_json must not be called for a non-US boundary")

    cfg = PipelineConfig(data_dir=tmp_path / "data")
    out = fetch_demographics_per_hex(NON_US_BOUNDARY, hex_gdf, cfg, fetch_json=fetch_json)

    assert calls == []  # the gate short-circuited before any HTTP call was made
    assert list(out.columns) == ["h3", "median_income", "pct_over_65", "pct_under_5"]
    assert len(out) == 0


def test_exception_in_fetch_json_after_gate_returns_empty_without_raising(tmp_path, hex_gdf):
    def boom(url, params):
        raise ConnectionError("network is down")

    cfg = PipelineConfig(data_dir=tmp_path / "data")
    # Must not raise — demographics.py contract: ANY exception -> WARNING + empty frame.
    out = fetch_demographics_per_hex(US_BOUNDARY, hex_gdf, cfg, fetch_json=boom)

    assert list(out.columns) == ["h3", "median_income", "pct_over_65", "pct_under_5"]
    assert len(out) == 0


def test_empty_frame_has_zero_rows_and_no_country_leak_for_lowercase_variant(tmp_path, hex_gdf):
    # Gate is case-insensitive ("united states" in country.lower()).
    boundary = _make_boundary("united states of america")
    cfg = PipelineConfig(data_dir=tmp_path / "data")
    fetch_json = _make_fake_fetch_json()

    out = fetch_demographics_per_hex(boundary, hex_gdf, cfg, fetch_json=fetch_json)
    assert len(fetch_json.calls) > 0  # this one *should* go through the gate
    assert len(out) > 0


# --------------------------------------------------------------------------------------
# Happy path: full US pipeline through areal interpolation
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demographics_result(hex_gdf, tmp_path_factory) -> pd.DataFrame:
    cfg = PipelineConfig(data_dir=tmp_path_factory.mktemp("demographics_data"))
    fetch_json = _make_fake_fetch_json()
    return fetch_demographics_per_hex(US_BOUNDARY, hex_gdf, cfg, fetch_json=fetch_json)


def test_result_columns_and_dtype(demographics_result):
    assert list(demographics_result.columns) == ["h3", "median_income", "pct_over_65", "pct_under_5"]


def test_all_hexes_present_since_every_hex_has_some_data(hex_cells, demographics_result):
    assert set(demographics_result["h3"]) == set(hex_cells)
    assert len(demographics_result) == len(hex_cells)


def test_hex_fully_inside_tract_a_matches_tract_a_exactly(hex_cells, demographics_result):
    row = demographics_result.set_index("h3").loc[hex_cells[0]]
    assert row["median_income"] == pytest.approx(INCOME_A)
    assert row["pct_over_65"] == pytest.approx(PCT_OVER_65_A)
    assert row["pct_under_5"] == pytest.approx(PCT_UNDER_5_A)


def test_hex_fully_inside_tract_b_sentinel_income_is_nan_but_pct_filled(hex_cells, demographics_result):
    row = demographics_result.set_index("h3").loc[hex_cells[2]]
    assert pd.isna(row["median_income"])  # tract B's income was the -666666666 sentinel
    assert row["pct_over_65"] == pytest.approx(PCT_OVER_65_B)
    assert row["pct_under_5"] == pytest.approx(PCT_UNDER_5_B)


def test_straddling_hex_is_area_weighted_mix(hex_cells, hex_gdf, demographics_result):
    hex1_id = hex_cells[1]
    hex1_geom_utm = hex_gdf.to_crs(UTM_CRS).set_index("h3").loc[hex1_id, "geometry"]

    # Ground truth from the *exact* round-tripped-through-4326 tract geometries (matching what
    # the implementation itself intersects against) rather than the pre-roundtrip UTM boxes.
    tract_a_rt = gpd.GeoSeries([_TRACT_A_UTM], crs=UTM_CRS).to_crs("EPSG:4326").to_crs(UTM_CRS).iloc[0]
    tract_b_rt = gpd.GeoSeries([_TRACT_B_UTM], crs=UTM_CRS).to_crs("EPSG:4326").to_crs(UTM_CRS).iloc[0]
    area_a = hex1_geom_utm.intersection(tract_a_rt).area
    area_b = hex1_geom_utm.intersection(tract_b_rt).area
    assert area_a > 0 and area_b > 0  # sanity: this hex really does straddle the tract line

    expected_pct_over_65 = (area_a * PCT_OVER_65_A + area_b * PCT_OVER_65_B) / (area_a + area_b)
    expected_pct_under_5 = (area_a * PCT_UNDER_5_A + area_b * PCT_UNDER_5_B) / (area_a + area_b)

    row = demographics_result.set_index("h3").loc[hex1_id]
    assert row["pct_over_65"] == pytest.approx(expected_pct_over_65, rel=1e-3)
    assert row["pct_under_5"] == pytest.approx(expected_pct_under_5, rel=1e-3)
    # Tract B's income was scrubbed to NaN, so it contributes zero *weight* to this column
    # specifically (per-column NaN exclusion) — hex1's income is 100% tract A's, not a mix.
    assert row["median_income"] == pytest.approx(INCOME_A)


# --------------------------------------------------------------------------------------
# Private helpers in isolation
# --------------------------------------------------------------------------------------


def test_find_tract_layer_id_picks_first_census_tracts_layer():
    layer_list = {
        "layers": [
            {"id": 5, "name": "Some Other Layer"},
            {"id": 8, "name": "Census Tracts"},
            {"id": 9, "name": "Census Tracts (detailed)"},
        ]
    }
    assert _find_tract_layer_id(lambda url, params: layer_list) == 8


def test_find_tract_layer_id_raises_when_absent():
    layer_list = {"layers": [{"id": 5, "name": "Census Blocks"}]}
    with pytest.raises(Exception):
        _find_tract_layer_id(lambda url, params: layer_list)


def test_query_acs_falls_back_to_older_vintage_when_newer_has_no_rows():
    calls = []

    def fetch_json(url, params):
        calls.append(url)
        if "/2023/" in url:
            return [_acs_header()]  # header only -> "no data rows"
        if "/2022/" in url:
            return ACS_ROWS
        raise AssertionError(url)

    rows = _query_acs(fetch_json, "12", "345")

    assert rows == ACS_ROWS
    assert any("/2023/" in u for u in calls)
    assert any("/2022/" in u for u in calls)


def test_query_acs_returns_none_when_every_vintage_fails():
    def fetch_json(url, params):
        raise TimeoutError("census.gov unreachable")

    assert _query_acs(fetch_json, "12", "345") is None


def test_fetch_tracts_paginates_until_transfer_limit_not_exceeded():
    feat_a = _tract_feature(GEOID_A, _TRACT_A_UTM)
    feat_b = _tract_feature(GEOID_B, _TRACT_B_UTM)
    pages = [
        {"type": "FeatureCollection", "features": [feat_a], "exceededTransferLimit": True},
        {"type": "FeatureCollection", "features": [feat_b], "exceededTransferLimit": False},
    ]
    offsets_requested: list[int] = []

    def fetch_json(url, params):
        offsets_requested.append(params["resultOffset"])
        return pages[len(offsets_requested) - 1]

    tracts = _fetch_tracts(fetch_json, 0, (3.6, 51.0, 3.8, 51.2))

    assert offsets_requested == [0, 1]  # second page starts where the first page's 1 feature ended
    assert sorted(tracts["GEOID"]) == sorted([GEOID_A, GEOID_B])
    assert tracts.crs.to_epsg() == 4326


def test_fetch_tracts_empty_result_is_empty_geodataframe():
    def fetch_json(url, params):
        return {"type": "FeatureCollection", "features": []}

    tracts = _fetch_tracts(fetch_json, 0, (3.6, 51.0, 3.8, 51.2))
    assert len(tracts) == 0
    assert list(tracts.columns) == ["GEOID", "geometry"]
