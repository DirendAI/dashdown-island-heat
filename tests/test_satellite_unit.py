"""Offline unit tests for heat_island.satellite pure helpers (no network, no STAC).

Only the numeric/logic core is exercised — masking, the Sentinel-2 baseline offset, the
spectral-index math, and item cloud-sorting. The network paths (STAC search, odc.stac.load,
.compute) are deliberately not tested here; they are covered by the runtime acceptance run.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from heat_island.satellite import (
    _indices_from_bands,
    _lst_from_dn,
    _s2_offset_for_times,
    _safe_ratio,
    _sort_and_cap_items,
)

# Landsat C2 ST scaling, mirrored here so the expectation is hand-derived, not imported.
_SCALE, _OFF_K, _KELVIN = 0.00341802, 149.0, 273.15


def _expected_lst(dn: float) -> float:
    return dn * _SCALE + _OFF_K - _KELVIN


# --- _lst_from_dn ------------------------------------------------------------


def test_lst_from_dn_valid_value_and_range_masking():
    # 45000 -> ~29.66 °C (in range); 0 -> nodata; 60000 -> ~80.9 °C (>70); 25000 -> ~-38.7 °C (<-20)
    dn = xr.DataArray(
        np.array([[45000.0, 0.0], [60000.0, 25000.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [51.0, 51.1], "longitude": [3.7, 3.8]},
    )
    qa = xr.DataArray(
        np.zeros((2, 2), dtype="uint16"),
        dims=("latitude", "longitude"),
        coords={"latitude": [51.0, 51.1], "longitude": [3.7, 3.8]},
    )

    out = _lst_from_dn(dn, qa)

    # coords/dims must survive — the pipeline needs lat/lon for hex aggregation downstream
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("latitude", "longitude")

    vals = out.values
    assert vals[0, 0] == pytest.approx(_expected_lst(45000.0))  # ~29.66 °C, exact formula
    assert np.isnan(vals[0, 1])  # DN == 0 -> nodata
    assert np.isnan(vals[1, 0])  # 60000 -> ~80.9 °C, above 70 °C cap
    assert np.isnan(vals[1, 1])  # 25000 -> ~-38.7 °C, below -20 °C floor


def test_lst_from_dn_qa_bit_masking():
    # QA bits 0-4 (mask 0b11111) drop a pixel; bits >= 5 must be ignored.
    dn = xr.DataArray(np.full((1, 3), 45000.0), dims=("latitude", "longitude"))
    qa = xr.DataArray(
        np.array([[0, 8, 64]], dtype="uint16"), dims=("latitude", "longitude")
    )  # 0 clear | 8 = cloud (bit 3) | 64 = bit 6 only

    out = _lst_from_dn(dn, qa).values

    assert out[0, 0] == pytest.approx(_expected_lst(45000.0))  # clear -> valid
    assert np.isnan(out[0, 1])  # cloud bit within 0-4 -> masked
    assert out[0, 2] == pytest.approx(_expected_lst(45000.0))  # high bit outside mask -> kept


# --- _s2_offset_for_times ----------------------------------------------------


def test_s2_offset_before_and_after_baseline_cutoff():
    times = np.array(
        ["2021-07-01", "2022-01-24", "2022-01-25", "2023-08-15"], dtype="datetime64[ns]"
    )
    # 2022-01-25 is the baseline-04.00 cutoff and counts as "new" (>= is inclusive).
    assert list(_s2_offset_for_times(times)) == [0.0, 0.0, -1000.0, -1000.0]


def test_s2_offset_accepts_datetime_like_list():
    assert list(_s2_offset_for_times(["2020-06-01", "2025-07-01"])) == [0.0, -1000.0]


# --- _indices_from_bands / _safe_ratio ---------------------------------------


def test_indices_from_bands_values_and_zero_denominator():
    #                     in-range pixel      | all-zero pixel (zero denominators)
    b02 = np.array([0.10, 0.10])
    b03 = np.array([0.20, 0.00])
    b04 = np.array([0.15, 0.00])
    b08 = np.array([0.50, 0.00])
    b11 = np.array([0.30, 0.00])

    ndvi, ndbi, ndwi, albedo = _indices_from_bands(b02, b03, b04, b08, b11)

    assert ndvi[0] == pytest.approx((0.50 - 0.15) / (0.50 + 0.15))
    assert ndbi[0] == pytest.approx((0.30 - 0.50) / (0.30 + 0.50))
    assert ndwi[0] == pytest.approx((0.20 - 0.50) / (0.20 + 0.50))
    assert albedo[0] == pytest.approx((0.10 + 0.20 + 0.15) / 3.0)

    # zero denominators -> NaN for the three normalized-difference indices
    assert np.isnan(ndvi[1]) and np.isnan(ndbi[1]) and np.isnan(ndwi[1])
    # albedo is an unguarded mean, so it still computes: (0.10 + 0 + 0) / 3
    assert albedo[1] == pytest.approx(0.10 / 3.0)


def test_indices_preserve_xarray_coords():
    coords = {"latitude": [51.0, 51.1], "longitude": [3.7, 3.8]}

    def band(v: float) -> xr.DataArray:
        return xr.DataArray(np.full((2, 2), v), dims=("latitude", "longitude"), coords=coords)

    ndvi, ndbi, ndwi, albedo = _indices_from_bands(
        band(0.10), band(0.20), band(0.15), band(0.50), band(0.30)
    )

    for arr in (ndvi, ndbi, ndwi, albedo):
        assert isinstance(arr, xr.DataArray)
        assert arr.dims == ("latitude", "longitude")
    assert float(ndvi.isel(latitude=0, longitude=0)) == pytest.approx((0.50 - 0.15) / (0.50 + 0.15))


def test_safe_ratio_guards_small_denominator():
    out = np.asarray(_safe_ratio(np.array([1.0, 1.0, 1.0]), np.array([2.0, 0.0, 1e-9])))
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])  # exact zero denominator
    assert np.isnan(out[2])  # |den| < 1e-6


# --- _sort_and_cap_items -----------------------------------------------------


class _FakeItem:
    """Minimal stand-in exposing the ``.properties`` dict that _sort_and_cap_items reads."""

    def __init__(self, cloud):
        self.properties = {} if cloud is None else {"eo:cloud_cover": cloud}


def test_sort_and_cap_items_orders_by_cloud_and_caps():
    items = [_FakeItem(50), _FakeItem(None), _FakeItem(5), _FakeItem(20)]
    out = _sort_and_cap_items(items, max_items=3)
    assert [it.properties.get("eo:cloud_cover") for it in out] == [5, 20, 50]
    assert len(out) == 3  # the missing-cloud item (-> 100) is dropped by the cap


def test_sort_and_cap_items_missing_cloud_treated_as_100():
    out = _sort_and_cap_items([_FakeItem(None), _FakeItem(90)], max_items=5)
    assert out[0].properties.get("eo:cloud_cover") == 90  # 90 sorts before missing (100)
    assert out[1].properties.get("eo:cloud_cover") is None


# --- land cover (fetch_landcover_per_hex) pure logic -------------------------
#
# ESA WorldCover reclassification, tree indicator, version selection and the class->fraction->hex
# aggregation round trip. All offline (synthetic arrays / fake STAC items), no network.

import datetime as _dt  # noqa: E402

import h3  # noqa: E402
import pandas as pd  # noqa: E402

from heat_island.config import LANDCOVER_TREE_CLASS, PLANTABLE_CLASS_WEIGHTS  # noqa: E402
from heat_island.hexgrid import points_to_hex_means  # noqa: E402
from heat_island.satellite import (  # noqa: E402
    _latest_version_items,
    _parse_major_version,
    _plantable_weights_lookup,
    _tree_indicator,
)


# --- _plantable_weights_lookup -----------------------------------------------


def test_plantable_weights_lookup_known_unknown_and_nodata():
    # every catalogued class + an unknown class (15) + nodata (0)
    classes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 15, 0], dtype="uint8")
    out = np.asarray(_plantable_weights_lookup(classes))

    for i, code in enumerate(classes.tolist()):
        if code == 0:
            assert np.isnan(out[i])  # nodata -> NaN
        elif code == 15:
            assert out[i] == pytest.approx(0.0)  # unknown class -> 0.0
        else:
            assert out[i] == pytest.approx(PLANTABLE_CLASS_WEIGHTS[code])
    # the anchors the architect called out explicitly
    assert out[4] == pytest.approx(0.15)  # 50 built-up -> tree-pit/depave credit
    assert out[0] == pytest.approx(0.0)   # 10 tree cover -> no ADDITIONAL room
    assert out[1] == pytest.approx(1.0)   # 20 shrubland -> fully plantable


def test_plantable_weights_lookup_preserves_xarray_coords():
    coords = {"latitude": [51.0, 51.1], "longitude": [3.7, 3.8]}
    classes = xr.DataArray(
        np.array([[10, 30], [50, 0]], dtype="uint8"), dims=("latitude", "longitude"), coords=coords
    )

    out = _plantable_weights_lookup(classes)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("latitude", "longitude")
    assert float(out.sel(latitude=51.0, longitude=3.7)) == pytest.approx(0.0)   # tree cover
    assert float(out.sel(latitude=51.0, longitude=3.8)) == pytest.approx(1.0)   # grassland
    assert float(out.sel(latitude=51.1, longitude=3.7)) == pytest.approx(0.15)  # built-up
    assert np.isnan(float(out.sel(latitude=51.1, longitude=3.8)))               # nodata


# --- _tree_indicator ---------------------------------------------------------


def test_tree_indicator_only_class_10_and_nodata():
    classes = np.array([LANDCOVER_TREE_CLASS, 20, 50, 80, 0], dtype="uint8")
    out = np.asarray(_tree_indicator(classes))
    assert out[0] == pytest.approx(1.0)  # tree cover -> 1
    assert out[1] == pytest.approx(0.0)  # shrubland -> 0
    assert out[2] == pytest.approx(0.0)  # built-up -> 0
    assert out[3] == pytest.approx(0.0)  # water -> 0
    assert np.isnan(out[4])              # nodata -> NaN


# --- class -> fraction -> points_to_hex_means round trip ---------------------


def test_landcover_fractions_aggregate_per_hex_with_nodata_skipped():
    res = 9
    # two well-separated res-9 cells; place pixels at each cell centre so membership is exact
    cell_a = h3.latlng_to_cell(51.05, 3.72, res)
    cell_b = h3.latlng_to_cell(51.20, 3.90, res)
    lat_a, lon_a = h3.cell_to_latlng(cell_a)
    lat_b, lon_b = h3.cell_to_latlng(cell_b)

    classes = np.array([30, 10, 50, 0], dtype="uint8")  # A: grass+tree ; B: built + nodata
    df = pd.DataFrame({"lat": [lat_a, lat_a, lat_b, lat_b], "lon": [lon_a, lon_a, lon_b, lon_b]})
    df["plantable_fraction"] = np.asarray(_plantable_weights_lookup(classes))
    df["tree_fraction"] = np.asarray(_tree_indicator(classes))

    out = points_to_hex_means(df, res, ["plantable_fraction", "tree_fraction"]).set_index("h3")

    # hex A: grassland(plantable 1.0, tree 0) + tree cover(plantable 0.0, tree 1) -> means 0.5/0.5
    assert out.loc[cell_a, "plantable_fraction"] == pytest.approx(0.5)
    assert out.loc[cell_a, "tree_fraction"] == pytest.approx(0.5)
    # hex B: built-up(0.15) + nodata(NaN, skipped by the skipna mean) -> plantable 0.15, tree 0.0
    assert out.loc[cell_b, "plantable_fraction"] == pytest.approx(0.15)
    assert out.loc[cell_b, "tree_fraction"] == pytest.approx(0.0)
    # both fractions land in [0, 1]
    assert out[["plantable_fraction", "tree_fraction"]].to_numpy().min() >= 0.0
    assert out[["plantable_fraction", "tree_fraction"]].to_numpy().max() <= 1.0


# --- version selection -------------------------------------------------------


def test_parse_major_version():
    assert _parse_major_version("2.0.0") == 2
    assert _parse_major_version("1.0.0") == 1
    assert _parse_major_version("10.2.3") == 10  # int parse, not lexical ("10" < "9" lexically)
    assert _parse_major_version(None) is None
    assert _parse_major_version("garbage") is None


class _FakeVersionItem:
    """Stand-in STAC item exposing ``.properties`` and ``.datetime`` for version selection."""

    def __init__(self, version=None, dt=None, start=None):
        props = {}
        if version is not None:
            props["esa_worldcover:product_version"] = version
        if start is not None:
            props["start_datetime"] = start
        self.properties = props
        self.datetime = dt
        self.id = f"item-v{version}"


def test_latest_version_items_keeps_newest_major():
    items = [_FakeVersionItem("1.0.0"), _FakeVersionItem("2.0.0"), _FakeVersionItem("1.0.0")]
    out = _latest_version_items(items)
    assert [it.properties["esa_worldcover:product_version"] for it in out] == ["2.0.0"]


def test_latest_version_items_datetime_fallback_when_property_absent():
    # no version property anywhere -> fall back to the latest item time (WorldCover uses ranges)
    old = _FakeVersionItem(dt=_dt.datetime(2020, 1, 1))
    new = _FakeVersionItem(dt=_dt.datetime(2021, 1, 1))
    assert _latest_version_items([old, new]) == [new]
