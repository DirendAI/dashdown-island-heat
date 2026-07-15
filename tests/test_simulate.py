"""Offline tests for heat_island.simulate — greening counterfactual + priority scoring.

No network access: all data comes from tests/synth.make_city (pure numpy + h3, deterministic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heat_island.config import PipelineConfig
from heat_island.model import train_and_evaluate
from heat_island.simulate import compute_priority, greening_target_ndvi, run_greening

from tests.synth import make_city


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture(scope="module")
def city_df(cfg):
    # Default ndvi_scale=0.8 yields >20 park hexes (is_park = ndvi > 0.7), well above
    # cfg.min_park_hexes, so greening_target_ndvi should take the park-quantile branch.
    return make_city(seed=42)


@pytest.fixture(scope="module")
def fitted_model(city_df, cfg):
    return train_and_evaluate(city_df, cfg).model


@pytest.fixture(scope="module")
def greened(city_df, cfg, fitted_model):
    return run_greening(city_df, fitted_model, cfg)


# --------------------------------------------------------------------------------------
# greening_target_ndvi
# --------------------------------------------------------------------------------------


def test_greening_target_uses_park_quantile(city_df, cfg):
    park_count = int(city_df["is_park"].sum())
    assert park_count >= cfg.min_park_hexes  # sanity: default city has enough parks

    target = greening_target_ndvi(city_df, cfg)
    expected = city_df.loc[city_df["is_park"], "ndvi"].quantile(cfg.park_ndvi_quantile)
    assert target == pytest.approx(expected)


def test_greening_target_fallback_when_too_few_parks(cfg):
    low_park_df = make_city(seed=42, ndvi_scale=0.25)
    park_count = int(low_park_df["is_park"].sum())
    assert park_count < cfg.min_park_hexes  # sanity: this variant has too few parks

    target = greening_target_ndvi(low_park_df, cfg)
    expected = low_park_df["ndvi"].quantile(cfg.fallback_ndvi_quantile)
    assert target == pytest.approx(expected)


def test_greening_target_fallback_when_is_park_column_absent(city_df, cfg):
    # is_park may legitimately be absent (simulate.py must not crash on df.get(...) semantics).
    df_no_is_park = city_df.drop(columns=["is_park"])
    target = greening_target_ndvi(df_no_is_park, cfg)
    expected = df_no_is_park["ndvi"].quantile(cfg.fallback_ndvi_quantile)
    assert target == pytest.approx(expected)


# --------------------------------------------------------------------------------------
# run_greening
# --------------------------------------------------------------------------------------


def test_run_greening_adds_expected_columns(greened):
    assert "predicted_lst_c" in greened.columns
    assert "predicted_cooling_c" in greened.columns


def test_run_greening_cooling_nonnegative_everywhere(greened):
    assert (greened["predicted_cooling_c"] >= 0).all()


def test_run_greening_already_green_hexes_get_no_cooling(city_df, cfg, greened):
    target = greening_target_ndvi(city_df, cfg)
    already_green = greened[greened["ndvi"] >= target]

    assert len(already_green) > 0
    assert already_green["predicted_cooling_c"].mean() < 0.15
    # Their NDVI is unchanged by max(ndvi, target), so X_cf == X exactly for these rows and
    # predicted_lst_cf == predicted_lst_c bit-for-bit -> cooling is exactly 0, not just small.
    assert np.allclose(already_green["predicted_cooling_c"], 0.0, atol=1e-6)


def test_run_greening_low_ndvi_hexes_get_positive_cooling(city_df, cfg, greened):
    target = greening_target_ndvi(city_df, cfg)
    low_ndvi = greened[greened["ndvi"] < target - 0.1]

    assert len(low_ndvi) > 0
    assert low_ndvi["predicted_cooling_c"].mean() > 0


# --------------------------------------------------------------------------------------
# compute_priority
# --------------------------------------------------------------------------------------


def test_compute_priority_bounds_and_max(greened):
    prioritized = compute_priority(greened)

    assert "priority_score" in prioritized.columns
    scores = prioritized["priority_score"]
    assert scores.notna().all()  # mean_lst_c is never NaN in this synthetic city
    assert (scores >= 0).all()
    assert (scores <= 1).all()
    assert scores.max() == pytest.approx(1.0)


def test_compute_priority_no_demographics_matches_all_nan_demographics(greened):
    no_demo = compute_priority(greened)  # no demographic columns at all

    all_nan_demo = greened.copy()
    for col in ("median_income", "pct_over_65", "pct_under_5"):
        all_nan_demo[col] = np.nan
    with_nan_demo = compute_priority(all_nan_demo)

    assert np.allclose(
        no_demo["priority_score"].to_numpy(), with_nan_demo["priority_score"].to_numpy()
    )


def test_compute_priority_low_income_outranks_high_income():
    # Isolated unit test of compute_priority's vulnerability term: compute_priority only needs
    # mean_lst_c + predicted_cooling_c (+ demographics) so a plausible synthetic
    # predicted_cooling_c column is used directly, no model fit required.
    df = make_city(seed=42, with_demographics=True)
    rng = np.random.default_rng(0)
    df["predicted_cooling_c"] = rng.uniform(0.0, 3.0, len(df))

    base = df.iloc[[0]].copy()

    poor = base.copy()
    poor["median_income"] = 20_000.0  # below the whole city's income range -> highest vulnerability
    poor["pct_over_65"] = 20.0
    poor["pct_under_5"] = 10.0
    poor["predicted_cooling_c"] = 2.0  # identical to `rich` so only vulnerability can differ

    rich = base.copy()
    rich["median_income"] = 150_000.0  # above the whole city's income range -> lowest vulnerability
    rich["pct_over_65"] = 20.0
    rich["pct_under_5"] = 10.0
    rich["predicted_cooling_c"] = 2.0

    combined = pd.concat([df, poor, rich], ignore_index=True)
    prioritized = compute_priority(combined)

    poor_score = prioritized.iloc[-2]["priority_score"]
    rich_score = prioritized.iloc[-1]["priority_score"]

    # heat_rank and cooling_norm are identical between poor/rich by construction (same
    # mean_lst_c, same predicted_cooling_c); pct_over_65/pct_under_5 also tie, so the
    # strictly lower income must strictly win on vulnerability alone.
    assert poor_score > rich_score
