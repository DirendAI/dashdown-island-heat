"""Assemble the per-hex feature table for one city.

Joins the canonical H3 grid with satellite (LST, Sentinel-2 indices, DEM), OSM and
optional demographic features, applies the fill rules from ARCHITECTURE.md, and returns
one GeoDataFrame ready for modelling.
"""

from __future__ import annotations

import geopandas as gpd

from .boundary import CityBoundary
from .config import DEMOGRAPHIC_COLS, FEATURES, PipelineConfig
from .demographics import fetch_demographics_per_hex
from .hexgrid import hexes_to_gdf, polygon_to_hexes
from .osm_features import fetch_osm_features_per_hex
from .satellite import (
    fetch_elevation_per_hex,
    fetch_landcover_per_hex,
    fetch_lst_per_hex,
    fetch_s2_indices_per_hex,
)
from .util import PipelineError, get_logger

log = get_logger(__name__)


def build_feature_table(boundary: CityBoundary, cfg: PipelineConfig) -> gpd.GeoDataFrame:
    """Canonical hex grid for the city with all model features + target attached."""
    cells = polygon_to_hexes(boundary.geometry, cfg.h3_resolution)
    grid = hexes_to_gdf(cells)
    log.info("%s: %d hexes at H3 res %d", boundary.name, len(grid), cfg.h3_resolution)

    lst = fetch_lst_per_hex(boundary, cfg)
    s2 = fetch_s2_indices_per_hex(boundary, cfg)
    dem = fetch_elevation_per_hex(boundary, cfg)
    try:
        landcover = fetch_landcover_per_hex(boundary, cfg)
    except Exception as exc:  # plantability degrades gracefully to the unconstrained model
        log.warning(
            "%s: land-cover fetch failed (%s) — plantable_fraction defaults to 1.0 "
            "(unconstrained greening) and tree_fraction to 0.0",
            boundary.name, exc,
        )
        landcover = None
    osm = fetch_osm_features_per_hex(boundary, grid, cfg)
    demo = fetch_demographics_per_hex(boundary, grid, cfg)

    df = grid
    parts = [lst, s2, dem, osm] if landcover is None else [lst, s2, dem, landcover, osm]
    for part in parts:
        df = df.merge(part, on="h3", how="left")
    if len(demo):
        df = df.merge(demo, on="h3", how="left")
    for col in DEMOGRAPHIC_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    # Hexes without satellite coverage cannot be modelled.
    n0 = len(df)
    df = df[df["mean_lst_c"].notna() & df["ndvi"].notna()].copy()
    dropped = n0 - len(df)
    if dropped:
        log.warning("%s: dropped %d/%d hexes lacking LST or NDVI coverage", boundary.name, dropped, n0)

    # Fill rules (ARCHITECTURE.md).
    df["elevation"] = df["elevation"].fillna(df["elevation"].median())
    for col in ("dist_water_m", "dist_park_m"):
        df[col] = df[col].fillna(cfg.dist_cap_m).clip(upper=cfg.dist_cap_m)
    for col in ("building_density", "road_density"):
        df[col] = df[col].fillna(0.0)
    for col in ("ndbi", "ndwi", "albedo"):
        df[col] = df[col].fillna(df[col].median())
    if "plantable_fraction" in df.columns:
        df["plantable_fraction"] = df["plantable_fraction"].fillna(df["plantable_fraction"].median())
        df["tree_fraction"] = df["tree_fraction"].fillna(df["tree_fraction"].median())
    else:  # land-cover fetch failed entirely: unconstrained legacy behaviour
        df["plantable_fraction"] = 1.0
        df["tree_fraction"] = 0.0
    df["is_park"] = df["is_park"].fillna(False).astype(bool)

    if len(df) < cfg.min_hexes:
        raise PipelineError(
            f"Only {len(df)} usable hexes for '{boundary.name}' (need ≥ {cfg.min_hexes}). "
            "The city may be too small for this H3 resolution, or satellite coverage is missing — "
            "try a lower --resolution or a larger administrative area."
        )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:  # contract violation upstream — fail loudly
        raise PipelineError(f"Feature columns missing after assembly: {missing}")

    ordered = [
        "h3", "lat", "lon", "geometry", "is_park", *FEATURES,
        "plantable_fraction", "tree_fraction", "mean_lst_c", *DEMOGRAPHIC_COLS,
    ]
    df = df[ordered]
    log.info(
        "%s: feature table ready — %d hexes | LST %.1f–%.1f °C (mean %.1f) | NDVI mean %.2f "
        "| plantable mean %.2f | tree cover mean %.2f",
        boundary.name, len(df),
        df["mean_lst_c"].min(), df["mean_lst_c"].max(), df["mean_lst_c"].mean(),
        df["ndvi"].mean(), df["plantable_fraction"].mean(), df["tree_fraction"].mean(),
    )
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
