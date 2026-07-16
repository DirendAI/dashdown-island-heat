"""Synthetic offline city dataframe shared by test_model.py and test_simulate.py.

Not a pytest file itself (no test_ prefix) — a plain importable helper, per ARCHITECTURE.md's
"no network access in unit tests" rule. Everything here is pure numpy + h3 (h3 is a local C
library with no network calls), so it is fully offline and deterministic given a seed.

We deliberately do NOT use tests/conftest.py for this: multiple agents are writing tests
concurrently and none of them owns a shared conftest, so a plain importable module avoids
collisions.
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd

# Roughly Ghent, Belgium — arbitrary but realistic for a mid-latitude European city.
LAT0 = 51.0
LON0 = 3.7
N_LAT = 25
N_LON = 24
SPACING = 0.004
RES = 9


def make_city(
    n_lat: int = N_LAT,
    n_lon: int = N_LON,
    lat0: float = LAT0,
    lon0: float = LON0,
    spacing: float = SPACING,
    res: int = RES,
    seed: int = 42,
    ndvi_scale: float = 0.8,
    with_demographics: bool = False,
    with_plantable: bool = False,
) -> pd.DataFrame:
    """~n_lat * n_lon synthetic H3 hexes with plausible features and a known-structure LST target.

    Columns: h3, lat, lon, config.FEATURES (ndvi, ndbi, ndwi, albedo, elevation,
    building_density, road_density, dist_water_m, dist_park_m), mean_lst_c, is_park, and
    (when with_demographics=True) median_income, pct_over_65, pct_under_5.

    with_plantable=True (default False, so existing callers are unaffected) additionally adds
    `plantable_fraction = clip(1.2 - ndbi - building_density, 0, 1)`: dense/built-up hexes
    (high ndbi + building_density) saturate at 0, open low-density hexes saturate at 1, and
    plenty of hexes land strictly in between — exercising all three plantability regimes in
    simulate.py's constrained counterfactual. Computed purely from already-drawn columns, so
    it does not consume extra RNG draws and never perturbs any other column's values.

    NDVI is a smooth-ish spatial pattern (sin/cos of grid indices) plus noise, so nearby hexes
    have correlated features/target — this matters for spatial-CV tests. `ndvi_scale` controls
    the amplitude of that pattern: the default (0.8) spreads NDVI roughly over [0.05, 0.85] and
    yields well over min_park_hexes(=20) hexes with ndvi > 0.7 ("is_park"); pass a low value
    (e.g. 0.2) to cap NDVI well below 0.7 everywhere, producing a city with too few (here: zero)
    park hexes, for testing simulate.py's NDVI-quantile fallback path.

    mean_lst_c has a known linear-ish structure (44 - 14*ndvi + 6*ndbi + 0.5*building_density
    - 0.001*elevation + N(0, 0.6)) so a trained model's R² is predictably strong (~0.9+ in
    practice with the default parameters) and ndvi dominates SHAP importance by construction.
    """
    rng = np.random.default_rng(seed)

    lat_idx = np.repeat(np.arange(n_lat), n_lon)
    lon_idx = np.tile(np.arange(n_lon), n_lat)
    lat = lat0 + lat_idx * spacing
    lon = lon0 + lon_idx * spacing
    cells = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(lat, lon)]

    grid = pd.DataFrame({"h3": cells, "lat": lat, "lon": lon, "_i": lat_idx, "_j": lon_idx})
    grid = grid.drop_duplicates(subset="h3").reset_index(drop=True)
    n = len(grid)
    i = grid["_i"].to_numpy(dtype=float)
    j = grid["_j"].to_numpy(dtype=float)

    # Smooth-ish spatial pattern in [0, 1] (min-max normalized sin/cos of grid position).
    phase = np.sin(i / 4.0) + np.cos(j / 3.5)
    phase_norm = (phase - phase.min()) / (phase.max() - phase.min())

    ndvi = 0.05 + ndvi_scale * phase_norm + rng.normal(0, 0.04, n)
    ndvi = np.clip(ndvi, 0.02, 0.9)

    ndbi = 0.6 - 0.5 * ndvi + rng.normal(0, 0.05, n)
    ndwi = np.clip(rng.normal(-0.1, 0.1, n), -1.0, 1.0)
    albedo = rng.uniform(0.10, 0.30, n)
    elevation = rng.uniform(0.0, 60.0, n)
    building_density = rng.uniform(0.0, 1.0, n)
    road_density = rng.uniform(0.0, 15.0, n)
    dist_water_m = rng.uniform(0.0, 10_000.0, n)
    dist_park_m = rng.uniform(0.0, 10_000.0, n)

    noise = rng.normal(0.0, 0.6, n)
    mean_lst_c = 44.0 - 14.0 * ndvi + 6.0 * ndbi + 0.5 * building_density - 0.001 * elevation + noise

    is_park = ndvi > 0.7

    df = pd.DataFrame(
        {
            "h3": grid["h3"].to_numpy(),
            "lat": grid["lat"].to_numpy(),
            "lon": grid["lon"].to_numpy(),
            "ndvi": ndvi,
            "ndbi": ndbi,
            "ndwi": ndwi,
            "albedo": albedo,
            "elevation": elevation,
            "building_density": building_density,
            "road_density": road_density,
            "dist_water_m": dist_water_m,
            "dist_park_m": dist_park_m,
            "mean_lst_c": mean_lst_c,
            "is_park": is_park,
        }
    )

    if with_demographics:
        df["median_income"] = rng.uniform(25_000.0, 120_000.0, n)
        df["pct_over_65"] = rng.uniform(5.0, 25.0, n)
        df["pct_under_5"] = rng.uniform(3.0, 12.0, n)

    if with_plantable:
        df["plantable_fraction"] = np.clip(1.2 - ndbi - building_density, 0.0, 1.0)

    return df
