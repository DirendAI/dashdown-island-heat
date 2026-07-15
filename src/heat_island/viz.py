"""Sanity-check plotting: hex grid + boundary overlay, saved as a PNG."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — must happen before pyplot import

import geopandas as gpd
import h3
import matplotlib.pyplot as plt

from .boundary import CityBoundary
from .util import get_logger

log = get_logger(__name__)


def plot_city_grid(
    boundary: CityBoundary,
    hex_gdf: gpd.GeoDataFrame,
    out_png: Path,
    value_col: str | None = None,
    title: str | None = None,
) -> Path:
    """Plot the hex grid (optionally choropleth by `value_col`) with the boundary outline on top.

    Saves a 150 dpi PNG at `out_png` and returns that path.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if len(hex_gdf):
        if value_col is not None and value_col in hex_gdf.columns:
            hex_gdf.plot(
                ax=ax,
                column=value_col,
                cmap="inferno",
                legend=True,
                edgecolor="white",
                linewidth=0.2,
                alpha=0.85,
            )
        else:
            hex_gdf.plot(ax=ax, color="steelblue", edgecolor="white", linewidth=0.2, alpha=0.5)

    gpd.GeoSeries([boundary.geometry], crs="EPSG:4326").plot(
        ax=ax, facecolor="none", edgecolor="black", linewidth=2
    )

    # Equal-area-looking aspect: 1 degree of longitude covers cos(lat) as much ground as
    # 1 degree of latitude, so compensate by that factor.
    ax.set_aspect(1 / math.cos(math.radians(boundary.centroid_lat)))

    if title is None:
        if len(hex_gdf):
            res = h3.get_resolution(hex_gdf["h3"].iloc[0])
            title = f"{boundary.name} — {len(hex_gdf)} hexes (res {res})"
        else:
            title = f"{boundary.name} — 0 hexes"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved city grid plot to %s (%d hexes)", out_png, len(hex_gdf))
    return out_png
