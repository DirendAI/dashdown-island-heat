"""Offline tests for heat_island.model — spatial-CV LightGBM training + SHAP.

No network access: all data comes from tests/synth.make_city (pure numpy + h3, deterministic).
"""

from __future__ import annotations

import h3
import lightgbm as lgb
import numpy as np
import pytest

from heat_island.config import FEATURES, PipelineConfig
from heat_island.model import ModelResult, train_and_evaluate
from heat_island.util import PipelineError

from tests.synth import make_city


@pytest.fixture(scope="module")
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture(scope="module")
def city_df(cfg):
    # Default ndvi_scale gives a strong, known ndvi/ndbi/building_density -> mean_lst_c
    # structure (see tests/synth.py), so the model should recover a high R².
    return make_city(seed=42)


def test_train_and_evaluate_quality(city_df, cfg):
    result = train_and_evaluate(city_df, cfg)

    assert isinstance(result, ModelResult)
    assert isinstance(result.model, lgb.LGBMRegressor)
    assert result.r2 > 0.6
    assert result.mae < 2.0
    assert result.n_train == len(city_df)


def test_shap_importance_keys_and_ndvi_dominance(city_df, cfg):
    result = train_and_evaluate(city_df, cfg)

    assert list(result.shap_importance.keys()) == FEATURES
    assert all(isinstance(v, float) for v in result.shap_importance.values())
    assert all(v >= 0 for v in result.shap_importance.values())

    # ndvi has by far the largest coefficient in the synthetic target (-14), so it should
    # dominate mean |SHAP| over every other feature.
    top_feature = max(result.shap_importance, key=result.shap_importance.get)
    assert top_feature == "ndvi"


def test_min_hexes_enforcement_raises(cfg):
    tiny_df = make_city(seed=42).head(10)
    assert len(tiny_df) < cfg.min_hexes

    with pytest.raises(PipelineError):
        train_and_evaluate(tiny_df, cfg)


def test_fold_models_length_matches_cv_splits_and_each_predicts(city_df, cfg):
    result = train_and_evaluate(city_df, cfg)

    # Mirror train_and_evaluate's own spatial-CV split-count logic (coarser H3 parent groups ->
    # GroupKFold(min(cv_folds, n_groups)), or plain KFold(2) if fewer than 2 groups) independently
    # via public h3 calls, rather than reaching into model.py's private _spatial_groups — this way
    # the assertion is a genuine "fold count matches CV splits used" contract check, not a
    # hardcoded magic number that happens to match today's synthetic city.
    groups = {
        h3.cell_to_parent(c, max(h3.get_resolution(c) - cfg.cv_parent_offset, 0)) for c in city_df["h3"]
    }
    expected_n_splits = min(cfg.cv_folds, len(groups)) if len(groups) >= 2 else 2

    assert len(result.fold_models) == expected_n_splits

    X = city_df[FEATURES]
    for fold_model in result.fold_models:
        assert isinstance(fold_model, lgb.LGBMRegressor)
        assert hasattr(fold_model, "predict")
        preds = fold_model.predict(X)
        assert len(preds) == len(city_df)
        assert np.all(np.isfinite(preds))


def test_determinism(city_df, cfg):
    result_a = train_and_evaluate(city_df, cfg)
    result_b = train_and_evaluate(city_df, cfg)

    assert result_a.r2 == result_b.r2
    assert result_a.mae == result_b.mae
    assert result_a.shap_importance == result_b.shap_importance

    # fold_models are refit from scratch each call (same fixed params + seed per fold split) --
    # both the count and every fold's predictions should be exactly reproducible too.
    assert len(result_a.fold_models) == len(result_b.fold_models)
    X = city_df[FEATURES]
    for fold_a, fold_b in zip(result_a.fold_models, result_b.fold_models):
        assert np.array_equal(fold_a.predict(X), fold_b.predict(X))
