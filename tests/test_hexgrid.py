"""Offline unit tests for heat_island.hexgrid (no network, no osmnx/Nominatim calls)."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from heat_island.hexgrid import (
    grid_stats,
    hex_parents,
    hexes_to_gdf,
    points_to_hex_means,
    polygon_to_hexes,
)

# A small box (~0.02 deg wide) around a point near Ghent, Belgium.
CENTER_LON, CENTER_LAT = 3.72, 51.05
HALF = 0.01
BOX = box(CENTER_LON - HALF, CENTER_LAT - HALF, CENTER_LON + HALF, CENTER_LAT + HALF)


@pytest.mark.parametrize("res", [8, 9])
def test_polygon_to_hexes_basic(res):
    cells = polygon_to_hexes(BOX, res)

    assert len(cells) > 0
    assert cells == sorted(cells)  # returned sorted
    assert len(cells) == len(set(cells))  # no duplicates
    for c in cells:
        assert h3.is_valid_cell(c)
        assert h3.get_resolution(c) == res


def test_hexes_to_gdf_geometry_valid_closed_and_within_box():
    cells = polygon_to_hexes(BOX, 9)
    gdf = hexes_to_gdf(cells)

    assert list(gdf.columns) == ["h3", "lat", "lon", "geometry"]
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == len(cells)

    assert gdf.geometry.is_valid.all()
    for geom in gdf.geometry:
        coords = list(geom.exterior.coords)
        assert coords[0] == coords[-1]  # ring is closed

    # hex cell centers must lie within the query box, padded a bit (cells straddling the
    # boundary have their *center* inside the box but can extend slightly beyond it).
    pad = HALF + 0.02
    assert gdf["lon"].between(CENTER_LON - pad, CENTER_LON + pad).all()
    assert gdf["lat"].between(CENTER_LAT - pad, CENTER_LAT + pad).all()


def test_hexes_to_gdf_roundtrip_centroids():
    cells = polygon_to_hexes(BOX, 9)
    gdf = hexes_to_gdf(cells)

    for _, row in gdf.iterrows():
        back = h3.latlng_to_cell(row["lat"], row["lon"], 9)
        assert back == row["h3"]


def test_points_to_hex_means_two_clusters_and_nan_drop():
    res = 9
    # Two clusters, each tightly clustered so every point in a cluster maps to the *same*
    # cell, far enough apart to land in different cells.
    c1_lat, c1_lon = CENTER_LAT - 0.004, CENTER_LON - 0.004
    c2_lat, c2_lon = CENTER_LAT + 0.004, CENTER_LON + 0.004
    jitter = 1e-7  # much smaller than a res-9 cell (~174 m edge)

    rows = [{"lat": c1_lat + i * jitter, "lon": c1_lon + i * jitter, "value": 10.0} for i in range(5)]
    rows += [{"lat": c2_lat + i * jitter, "lon": c2_lon + i * jitter, "value": 20.0} for i in range(5)]
    rows.append({"lat": np.nan, "lon": CENTER_LON, "value": 999.0})  # must be dropped
    rows.append({"lat": CENTER_LAT, "lon": np.nan, "value": 999.0})  # must be dropped
    df = pd.DataFrame(rows)

    out = points_to_hex_means(df, res, ["value"])

    assert set(out.columns) == {"h3", "value"}
    assert 999.0 not in out["value"].to_numpy()
    assert len(out) == 2  # exactly the two clusters survive

    cell1 = h3.latlng_to_cell(c1_lat, c1_lon, res)
    cell2 = h3.latlng_to_cell(c2_lat, c2_lon, res)
    means = out.set_index("h3")["value"]
    assert means.loc[cell1] == pytest.approx(10.0)
    assert means.loc[cell2] == pytest.approx(20.0)


def test_hex_parents_offset_math():
    cells = polygon_to_hexes(BOX, 9)
    parents = hex_parents(cells, 2)

    assert isinstance(parents, np.ndarray)
    assert len(parents) == len(cells)
    for cell, parent in zip(cells, parents):
        assert h3.get_resolution(parent) == 7
        assert h3.cell_to_parent(cell, 7) == parent

    # offset >= resolution clamps to res 0, not a negative resolution
    top_parents = hex_parents(cells, 99)
    for parent in top_parents:
        assert h3.get_resolution(parent) == 0


def test_polygon_to_hexes_empty_polygon_fallback():
    # A polygon far smaller than a single res-9 cell (~174 m edge) contains no cell center,
    # so h3shape_to_cells itself returns [] and polygon_to_hexes must fall back to
    # grid_disk(k=1) around the centroid — the point of this test is that the *public*
    # function never returns an empty list, regardless of that internal empty intersection.
    tiny = box(CENTER_LON, CENTER_LAT, CENTER_LON + 1e-6, CENTER_LAT + 1e-6)

    cells = polygon_to_hexes(tiny, 9)
    assert len(cells) > 0
    for c in cells:
        assert h3.is_valid_cell(c)
        assert h3.get_resolution(c) == 9


def test_grid_stats():
    cells = polygon_to_hexes(BOX, 9)
    stats = grid_stats(cells, BOX)

    assert set(stats.keys()) == {"n_hexes", "area_km2", "mean_hex_area_km2"}
    assert stats["n_hexes"] == len(cells)
    assert stats["area_km2"] > 0
    assert stats["mean_hex_area_km2"] > 0
    # a res-9 hex is roughly 0.1 km^2 — sanity bound, not an exact figure
    assert 0.01 < stats["mean_hex_area_km2"] < 1.0
