"""Offline tests for heat_island.simulate — greening counterfactual + priority scoring.

No network access: all data comes from tests/synth.make_city (pure numpy + h3, deterministic).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from heat_island.config import FEATURES, PipelineConfig
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
def trained(city_df, cfg):
    return train_and_evaluate(city_df, cfg)


@pytest.fixture(scope="module")
def fitted_model(trained):
    return trained.model


@pytest.fixture(scope="module")
def fold_models(trained):
    return trained.fold_models


@pytest.fixture(scope="module")
def greened(city_df, cfg, fitted_model):
    return run_greening(city_df, fitted_model, cfg)


@pytest.fixture(scope="module")
def city_df_plantable(cfg):
    # Same seed as city_df -> identical in every column except the added plantable_fraction
    # (computed from already-drawn ndbi/building_density columns, so it consumes no extra RNG
    # draws and does not perturb anything else). ndvi_scale default keeps the same park-quantile
    # greening target as city_df, so gap/cooling behaviour is directly comparable.
    return make_city(seed=42, with_plantable=True)


@pytest.fixture(scope="module")
def greened_plantable(city_df_plantable, cfg, fitted_model):
    # fitted_model was trained on city_df (without plantable_fraction), which is fine: it only
    # ever sees config.FEATURES, and plantable_fraction is deliberately not one of them.
    return run_greening(city_df_plantable, fitted_model, cfg)


@pytest.fixture(scope="module")
def half_split_models(city_df, cfg):
    """Two genuinely different fitted models, trained on disjoint halves of city_df.

    Used to exercise cooling_uncertainty_c's "different fold models -> nonzero spread" case,
    as opposed to passing the same model twice (which must give exactly zero spread).
    """
    shuffled = city_df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    mid = len(shuffled) // 2
    half_a = shuffled.iloc[:mid].reset_index(drop=True)
    half_b = shuffled.iloc[mid:].reset_index(drop=True)
    assert len(half_a) >= cfg.min_hexes and len(half_b) >= cfg.min_hexes
    return train_and_evaluate(half_a, cfg).model, train_and_evaluate(half_b, cfg).model


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
    assert "cooling_uncertainty_c" in greened.columns


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
# run_greening — plantability-constrained counterfactual
# --------------------------------------------------------------------------------------


def _unconstrained_cooling(model, df, cfg):
    """The pre-plantability `max(ndvi, target)` counterfactual, computed by hand for comparison."""
    target = greening_target_ndvi(df, cfg)
    X = df[FEATURES]
    X_cf = X.copy()
    X_cf["ndvi"] = np.maximum(X_cf["ndvi"], target)
    return np.clip(model.predict(X) - model.predict(X_cf), 0, None)


def test_plantable_zero_rows_get_exactly_zero_cooling(city_df_plantable, greened_plantable):
    zero_mask = city_df_plantable["plantable_fraction"] == 0
    assert zero_mask.sum() > 0  # sanity: the synthetic formula does saturate at 0 for some hexes

    zero_cooling = greened_plantable.loc[zero_mask, "predicted_cooling_c"]
    assert (zero_cooling == 0).all()


def test_plantable_one_rows_match_unconstrained_cooling(city_df_plantable, cfg, fitted_model, greened_plantable):
    one_mask = city_df_plantable["plantable_fraction"] == 1.0
    assert one_mask.sum() > 0  # sanity: the synthetic formula does saturate at 1 for some hexes

    manual_cooling = _unconstrained_cooling(fitted_model, city_df_plantable, cfg)
    got_cooling = greened_plantable.loc[one_mask, "predicted_cooling_c"].to_numpy()
    assert np.allclose(got_cooling, manual_cooling[one_mask.to_numpy()], atol=1e-9)


def test_plantable_intermediate_rows_get_intermediate_cooling(city_df_plantable, cfg, fitted_model, greened_plantable):
    plantable = city_df_plantable["plantable_fraction"]
    target = greening_target_ndvi(city_df_plantable, cfg)
    gap = np.clip(target - city_df_plantable["ndvi"], 0, None).to_numpy()
    manual_cooling = _unconstrained_cooling(fitted_model, city_df_plantable, cfg)

    # Only rows that actually have room to green (positive gap) AND where full plantability
    # would predict some cooling can meaningfully be "in between" zero and the unconstrained case.
    qualifying = ((plantable > 0) & (plantable < 1)).to_numpy() & (gap > 0) & (manual_cooling > 0)
    assert qualifying.sum() > 50  # plenty of hexes exercise this regime in the synthetic city

    actual = greened_plantable.loc[qualifying, "predicted_cooling_c"].to_numpy()
    unconstrained = manual_cooling[qualifying]

    # LightGBM predicts a step function of ndvi, so a small minority of rows can land in the
    # same leaf as the ndvi=0 or ndvi=target case (a tie, never an overshoot) rather than a
    # strictly interior value. Bound every row non-strictly, and require the strict inequality
    # (and the intended intermediate-cooling property) to hold for the large majority and on
    # average -- more informative than a single cherry-picked row and not flaky like a 100%
    # per-row strict assertion would be against a piecewise-constant model.
    assert (actual >= 0).all()
    assert (actual <= unconstrained + 1e-9).all()
    strictly_between = (actual > 0) & (actual < unconstrained)
    assert strictly_between.mean() > 0.9
    assert 0 < actual.mean() < unconstrained.mean()


def test_missing_plantable_fraction_matches_all_ones_and_warns(city_df, cfg, fitted_model, caplog):
    all_ones = city_df.copy()
    all_ones["plantable_fraction"] = 1.0

    with caplog.at_level(logging.WARNING, logger="heat_island.simulate"):
        missing = run_greening(city_df, fitted_model, cfg)
    ones = run_greening(all_ones, fitted_model, cfg)

    assert np.allclose(missing["predicted_lst_c"].to_numpy(), ones["predicted_lst_c"].to_numpy())
    assert np.allclose(missing["predicted_cooling_c"].to_numpy(), ones["predicted_cooling_c"].to_numpy())
    assert any("plantable" in record.message.lower() for record in caplog.records)


def test_plantable_fraction_nan_rows_treated_as_one(city_df_plantable, cfg, fitted_model):
    # A handful of per-row NaNs in an otherwise-present column, vs. those same rows pinned to
    # 1.0 explicitly (fillna(1.0) inside run_greening should make the two inputs equivalent).
    # Other rows' plantable_fraction is untouched and identical between the two frames, and
    # run_greening treats each row independently, so the whole output should match, not just
    # the perturbed rows.
    with_nan = city_df_plantable.copy()
    nan_idx = with_nan.index[with_nan["plantable_fraction"] > 0][:5]
    with_nan.loc[nan_idx, "plantable_fraction"] = np.nan

    forced_ones = with_nan.copy()
    forced_ones.loc[nan_idx, "plantable_fraction"] = 1.0

    got = run_greening(with_nan, fitted_model, cfg)
    expected = run_greening(forced_ones, fitted_model, cfg)

    assert np.allclose(got["predicted_cooling_c"].to_numpy(), expected["predicted_cooling_c"].to_numpy())


# --------------------------------------------------------------------------------------
# run_greening — cooling_uncertainty_c
# --------------------------------------------------------------------------------------


def test_cooling_uncertainty_nan_without_fold_models(greened, greened_plantable):
    assert greened["cooling_uncertainty_c"].isna().all()
    assert greened_plantable["cooling_uncertainty_c"].isna().all()


def test_cooling_uncertainty_zero_for_identical_fold_models(city_df_plantable, cfg, fitted_model):
    result = run_greening(city_df_plantable, fitted_model, cfg, fold_models=[fitted_model, fitted_model])

    assert result["cooling_uncertainty_c"].notna().all()
    assert np.allclose(result["cooling_uncertainty_c"].to_numpy(), 0.0, atol=1e-9)


def test_cooling_uncertainty_positive_for_different_fold_models(
    city_df_plantable, cfg, fitted_model, half_split_models
):
    model_a, model_b = half_split_models
    assert model_a is not model_b

    result = run_greening(city_df_plantable, fitted_model, cfg, fold_models=[model_a, model_b])
    uncertainty = result["cooling_uncertainty_c"]

    assert uncertainty.notna().all()
    assert (uncertainty >= 0).all()  # std is never negative
    assert (uncertainty > 0).any()  # genuinely different models must disagree somewhere


def test_cooling_uncertainty_uses_real_fold_models_from_train_and_evaluate(city_df_plantable, cfg, fitted_model, fold_models):
    # End-to-end: fold_models straight off ModelResult (not hand-built) plugged into run_greening.
    assert len(fold_models) >= 2
    result = run_greening(city_df_plantable, fitted_model, cfg, fold_models=fold_models)
    assert result["cooling_uncertainty_c"].notna().all()
    assert (result["cooling_uncertainty_c"] >= 0).all()


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


def test_compute_priority_unplantable_hottest_hex_scores_zero(city_df_plantable, greened_plantable):
    # No separate plantability multiplier exists in compute_priority (see its source comment) --
    # a zero-plantable hex must still score exactly 0 purely because run_greening already gave
    # it zero predicted_cooling_c, even when it is (artificially) made the single hottest hex.
    zero_idx = city_df_plantable.index[city_df_plantable["plantable_fraction"] == 0][0]

    forced = greened_plantable.copy()
    assert forced.loc[zero_idx, "predicted_cooling_c"] == 0  # from the zero-cooling test above
    forced.loc[zero_idx, "mean_lst_c"] = forced["mean_lst_c"].max() + 10.0  # now the hottest hex

    prioritized = compute_priority(forced)

    assert prioritized.loc[zero_idx, "mean_lst_c"] == forced["mean_lst_c"].max()  # still hottest
    assert prioritized.loc[zero_idx, "priority_score"] == 0
