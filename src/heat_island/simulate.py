"""Greening counterfactual simulation and tree-planting priority scoring.

City-agnostic: operates on a plain per-hex dataframe (h3 + config.FEATURES + mean_lst_c),
plus optionally `is_park` and the demographic columns from config.DEMOGRAPHIC_COLS when
present. Does not import boundary.py, db.py, satellite.py, osm_features.py or hexgrid.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEMOGRAPHIC_COLS, FEATURES, PipelineConfig
from .util import PipelineError, get_logger

log = get_logger(__name__)


def greening_target_ndvi(df: pd.DataFrame, cfg: PipelineConfig) -> float:
    """NDVI level hexes are "greened up" to in the counterfactual.

    75th percentile of NDVI among park hexes (`is_park`) if there are at least
    `cfg.min_park_hexes` of them; otherwise the 90th percentile of NDVI over all hexes
    (logged as a WARNING fallback — too few parks to set a local reference level).
    """
    if "is_park" in df.columns:
        is_park_mask = df["is_park"] == True  # noqa: E712 — explicit bool comparison, mirrors NA-safety of ARCHITECTURE.md's df.get(...) == True
    else:
        is_park_mask = pd.Series(False, index=df.index)

    park_ndvi = df.loc[is_park_mask, "ndvi"]
    if len(park_ndvi) >= cfg.min_park_hexes:
        target = float(park_ndvi.quantile(cfg.park_ndvi_quantile))
        log.info(
            "greening_target_ndvi: %.3f (%.0fth pct of %d park hexes)",
            target, cfg.park_ndvi_quantile * 100, len(park_ndvi),
        )
    else:
        target = float(df["ndvi"].quantile(cfg.fallback_ndvi_quantile))
        log.warning(
            "greening_target_ndvi: only %d park hexes (< min_park_hexes=%d) — falling back to "
            "%.0fth pct NDVI over all %d hexes = %.3f",
            len(park_ndvi), cfg.min_park_hexes, cfg.fallback_ndvi_quantile * 100, len(df), target,
        )

    if pd.isna(target):
        raise PipelineError(
            "Greening target NDVI is NaN — check the ndvi column for coverage before simulating."
        )
    return target


def run_greening(df: pd.DataFrame, model, cfg: PipelineConfig, fold_models=None) -> pd.DataFrame:
    """Add `predicted_lst_c`, `predicted_cooling_c`, `cooling_uncertainty_c` from a
    plantability-constrained greening counterfactual.

    predicted_lst_c: model prediction on the hex's observed features.

    Plantability-constrained counterfactual: a hex can only close the gap to the greening
    target (`greening_target_ndvi(df, cfg)`) in proportion to its plantable share.
    `gap = clip(target - ndvi, 0, None)` (never asks a hex to green past the target);
    `plantable_fraction` (0-1) scales how much of that gap is achievable — a hex with 0
    plantable share (water, solid existing canopy) gets `ndvi_cf == ndvi` (no change, hence
    zero cooling); a fully plantable hex reproduces the old unconstrained `max(ndvi, target)`
    behaviour. A `plantable_fraction` column absent entirely -> 1.0 everywhere (legacy
    unconstrained behaviour), logged as a WARNING; a per-row NaN within an existing column ->
    1.0 for that row only (no column-level warning, ARCHITECTURE.md's fillna leniency).
    `predicted_cooling_c` is the resulting drop in predicted LST, clipped at 0 so "greening"
    can never predict warming.

    cooling_uncertainty_c: the honest spread of the constrained-cooling estimate across the
    spatial-CV fold models (`fold_models`, from `ModelResult.fold_models`) — the std (ddof=0)
    of each fold model's *unclipped* `predict(X) - predict(X_cf)` delta. `fold_models` being
    None/empty (e.g. a caller that didn't keep them around) -> NaN column, not an error.
    """
    out = df.copy()
    target = greening_target_ndvi(df, cfg)

    X = out[FEATURES]
    predicted_lst_c = model.predict(X)

    ndvi = out["ndvi"]
    gap = np.clip(target - ndvi, 0, None)

    if "plantable_fraction" in out.columns:
        plantable = out["plantable_fraction"].fillna(1.0)
    else:
        log.warning(
            "run_greening: 'plantable_fraction' column absent — treating every hex as fully "
            "plantable (unconstrained greening, matches pre-plantability behaviour)"
        )
        plantable = pd.Series(1.0, index=out.index)

    ndvi_cf = ndvi + plantable * gap

    X_cf = X.copy()
    X_cf["ndvi"] = ndvi_cf
    predicted_lst_cf = model.predict(X_cf)

    predicted_cooling_c = np.clip(predicted_lst_c - predicted_lst_cf, 0, None)

    out["predicted_lst_c"] = predicted_lst_c
    out["predicted_cooling_c"] = predicted_cooling_c

    if fold_models:
        fold_deltas = np.stack(
            [np.asarray(fm.predict(X)) - np.asarray(fm.predict(X_cf)) for fm in fold_models],
            axis=0,
        )
        cooling_uncertainty_c = np.std(fold_deltas, axis=0, ddof=0)
        mean_uncertainty = float(np.mean(cooling_uncertainty_c))
    else:
        cooling_uncertainty_c = np.full(len(out), np.nan)
        mean_uncertainty = float("nan")

    out["cooling_uncertainty_c"] = cooling_uncertainty_c

    log.info(
        "run_greening: target NDVI=%.3f | predicted cooling max=%.2f°C mean=%.2f°C | "
        "cooling_uncertainty_c mean=%.3f°C (%d fold models)",
        target, float(np.max(predicted_cooling_c)), float(np.mean(predicted_cooling_c)),
        mean_uncertainty, len(fold_models) if fold_models else 0,
    )
    return out


def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Add `priority_score` in [0, 1] = heat rank x normalized cooling x vulnerability.

    heat_rank: percentile rank (0-1) of mean_lst_c — hotter hexes rank higher.
    cooling_norm: predicted_cooling_c scaled by its max (all-zero cooling -> all zeros).
    vulnerability: 1.0 when none of median_income/pct_over_65/pct_under_5 are available for a
    row (or the columns are absent entirely); otherwise 0.5 + 0.5 * (mean of the available
    ranked components: 1 - rank(income), rank(pct_over_65), rank(pct_under_5)) in [0.5, 1].
    priority_score = raw / raw.max() with raw = heat_rank * cooling_norm * vulnerability
    (raw.max() == 0 -> all zeros). NaN only where mean_lst_c itself was NaN.
    """
    out = df.copy()

    heat_rank = out["mean_lst_c"].rank(pct=True)

    cooling = out["predicted_cooling_c"]
    cooling_max = cooling.max()
    if pd.isna(cooling_max) or cooling_max == 0:
        log.warning("compute_priority: predicted_cooling_c is all-zero/NaN — cooling_norm set to 0")
        cooling_norm = pd.Series(0.0, index=out.index)
    else:
        cooling_norm = cooling / cooling_max

    present_demo_cols = [c for c in DEMOGRAPHIC_COLS if c in out.columns]
    if not present_demo_cols:
        vulnerability = pd.Series(1.0, index=out.index)
    else:
        components = pd.DataFrame(index=out.index)
        if "median_income" in present_demo_cols:
            components["median_income"] = 1 - out["median_income"].rank(pct=True)
        if "pct_over_65" in present_demo_cols:
            components["pct_over_65"] = out["pct_over_65"].rank(pct=True)
        if "pct_under_5" in present_demo_cols:
            components["pct_under_5"] = out["pct_under_5"].rank(pct=True)

        v_raw = components.mean(axis=1, skipna=True)
        vulnerability = 0.5 + 0.5 * v_raw
        all_missing = out[present_demo_cols].isna().all(axis=1)
        vulnerability[all_missing] = 1.0

    # Deliberately no separate plantability multiplier here: feasibility already enters through
    # the plantability-constrained cooling computed in run_greening (a zero-plantable hex has
    # zero predicted_cooling_c and therefore zero cooling_norm here); multiplying by
    # plantable_fraction again would double-penalize it.
    raw = heat_rank * cooling_norm * vulnerability
    raw_max = raw.max()
    if raw_max == 0:
        # All defined rows already sit at exactly 0 here; avoid a 0/0 division. Rows whose
        # mean_lst_c was NaN (and are therefore still NaN in `raw`) stay NaN, per contract.
        log.warning("compute_priority: raw priority score is uniformly zero across all hexes")
        priority_score = raw
    else:
        priority_score = raw / raw_max

    out["priority_score"] = priority_score
    return out
