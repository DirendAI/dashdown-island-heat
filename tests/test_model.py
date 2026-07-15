"""Offline tests for heat_island.model — spatial-CV LightGBM training + SHAP.

No network access: all data comes from tests/synth.make_city (pure numpy + h3, deterministic).
"""

from __future__ import annotations

import lightgbm as lgb
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


def test_determinism(city_df, cfg):
    result_a = train_and_evaluate(city_df, cfg)
    result_b = train_and_evaluate(city_df, cfg)

    assert result_a.r2 == result_b.r2
    assert result_a.mae == result_b.mae
    assert result_a.shap_importance == result_b.shap_importance
