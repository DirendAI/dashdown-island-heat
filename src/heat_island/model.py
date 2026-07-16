"""LightGBM model of hex-level land-surface temperature, with spatial CV + SHAP.

City-agnostic by design (see ARCHITECTURE.md): everything here operates on a plain per-hex
dataframe with an `h3` column, `config.FEATURES`, and `mean_lst_c`. No city_id or other
per-city logic lives in this module, so a future global multi-city model can reuse it as-is.

Decoupling note: this module intentionally does NOT import hexgrid.py (developed concurrently
by another agent). The one thing model.py needs from it — mapping each hex to a coarser
"parent" cell for spatial cross-validation blocking — is reimplemented locally in
`_spatial_groups` via h3 directly, exactly as specified in the architecture contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .config import FEATURES, PipelineConfig
from .util import PipelineError, get_logger

log = get_logger(__name__)


@dataclass
class ModelResult:
    """Trained LightGBM model plus its out-of-fold spatial-CV metrics and SHAP importances."""

    model: lgb.LGBMRegressor  # final model, refit on ALL rows
    r2: float  # out-of-fold spatial-CV R²
    mae: float  # out-of-fold MAE (°C)
    n_train: int
    shap_importance: dict[str, float]  # feature -> mean |SHAP|, in config.FEATURES order
    fold_models: list = field(default_factory=list)  # the k models trained on CV folds (fold
    # order) — kept so simulate.py can spread the greening-cooling estimate into an uncertainty


def _lgb_params(cfg: PipelineConfig) -> dict:
    """Fixed LightGBM hyperparameters (ARCHITECTURE.md pins these exactly)."""
    return dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,  # without this LightGBM never activates row bagging (subsample is inert)
        colsample_bytree=0.8,
        random_state=cfg.seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _spatial_groups(h3_cells: pd.Series, offset: int) -> np.ndarray:
    """Coarser H3 parent cell per row, used as the spatial-CV blocking key.

    Equivalent to the planned `hexgrid.hex_parents`, reimplemented locally per the
    decoupling rule (hexgrid.py is being written concurrently by another agent).
    """
    return np.array(
        [h3.cell_to_parent(c, max(h3.get_resolution(c) - offset, 0)) for c in h3_cells],
        dtype=object,
    )


def train_and_evaluate(df: pd.DataFrame, cfg: PipelineConfig) -> ModelResult:
    """Fit a LightGBM regressor of `mean_lst_c` on `config.FEATURES` with spatial-block CV.

    X = df[FEATURES] (NaNs allowed — LightGBM handles them natively; not imputed here).
    y = df["mean_lst_c"] (rows with NaN y are dropped before training/evaluation).
    Cross-validation is grouped by coarser H3 parent cells so folds don't leak nearby hexes
    into both train and test. Out-of-fold predictions are pooled across all folds to compute
    a single global R² / MAE. The returned model is refit on every valid row.
    """
    if len(df) < cfg.min_hexes:
        raise PipelineError(
            f"Only {len(df)} hexes available; need at least {cfg.min_hexes} to train a model."
        )

    y_full = df["mean_lst_c"].astype(float)
    valid = y_full.notna()
    if not valid.all():
        log.warning(
            "train_and_evaluate: dropping %d/%d rows with NaN mean_lst_c before training",
            int((~valid).sum()), len(df),
        )

    X = df.loc[valid, FEATURES].astype(float).reset_index(drop=True)
    y = y_full.loc[valid].reset_index(drop=True)
    h3_cells = df.loc[valid, "h3"].reset_index(drop=True)
    n_train = len(X)
    y_arr = y.to_numpy()

    groups = _spatial_groups(h3_cells, cfg.cv_parent_offset)
    n_groups = int(len(np.unique(groups)))

    if n_groups < 2:
        log.warning(
            "train_and_evaluate: only %d unique spatial-CV group(s) (cv_parent_offset=%d) — "
            "falling back to a plain shuffled KFold(2) instead of spatial GroupKFold",
            n_groups, cfg.cv_parent_offset,
        )
        splits = list(KFold(n_splits=2, shuffle=True, random_state=cfg.seed).split(X))
    else:
        n_splits = min(cfg.cv_folds, n_groups)
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    params = _lgb_params(cfg)
    oof_pred = np.full(n_train, np.nan)
    fold_models: list = []
    for train_idx, test_idx in splits:
        fold_model = lgb.LGBMRegressor(**params)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred[test_idx] = fold_model.predict(X.iloc[test_idx])
        fold_models.append(fold_model)

    covered = ~np.isnan(oof_pred)
    if not covered.all():
        log.warning(
            "train_and_evaluate: %d/%d rows never appeared in a CV test fold; excluding "
            "them from the out-of-fold R²/MAE", int((~covered).sum()), n_train,
        )
    r2 = float(r2_score(y_arr[covered], oof_pred[covered]))
    mae = float(mean_absolute_error(y_arr[covered], oof_pred[covered]))

    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)

    shap_values = shap.TreeExplainer(final_model).shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = {feat: float(val) for feat, val in zip(FEATURES, mean_abs_shap)}

    log.info(
        "train_and_evaluate: n_train=%d spatial_cv_groups=%d R²=%.3f MAE=%.3f°C",
        n_train, n_groups, r2, mae,
    )

    return ModelResult(
        model=final_model,
        r2=r2,
        mae=mae,
        n_train=n_train,
        shap_importance=shap_importance,
        fold_models=fold_models,
    )
