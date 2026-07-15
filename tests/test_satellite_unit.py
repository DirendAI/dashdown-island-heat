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
