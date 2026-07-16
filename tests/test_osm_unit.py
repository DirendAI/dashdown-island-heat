"""Offline unit tests for heat_island.osm_features's geometry helpers (no network, no Overpass).

Only the private, network-free math helpers are exercised here — `fetch_osm_features_per_hex`
itself calls osmnx/Overpass and is deliberately never invoked from a test (ARCHITECTURE.md: no
network access in unit tests). Everything is built directly in a local UTM frame (EPSG:32631,
the correct zone for ~3.7°E — the Ghent-area longitude used throughout the project) so
area/length/distance math is exact and independent of any reprojection error.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from heat_island.hexgrid import hexes_to_gdf
from heat_island.osm_features import (
    _building_density,
    _distance_to_nearest,
    _is_park,
    _road_density,
)

UTM_CRS = 32631  # correct UTM zone for ~3.7°E
LAT0, LON0 = 51.05, 3.72
RES = 9


def _hex_cluster(n: int = 6) -> list[str]:
    """`n` distinct res-9 H3 cells spaced out along a line near (LAT0, LON0)."""
    cells: list[str] = []
    seen: set[str] = set()
    i = 0
    while len(cells) < n:
        cell = h3.latlng_to_cell(LAT0, LON0 + i * 0.006, RES)
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)
        i += 1
    return cells


@pytest.fixture(scope="module")
def hex_cells() -> list[str]:
    return _hex_cluster(6)


@pytest.fixture(scope="module")
def hex_gdf_utm(hex_cells) -> gpd.GeoDataFrame:
    return hexes_to_gdf(hex_cells).to_crs(UTM_CRS)


@pytest.fixture(scope="module")
def hex_utm_by_h3(hex_gdf_utm) -> gpd.GeoDataFrame:
    return hex_gdf_utm.set_index("h3")


def _empty_utm_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], crs=UTM_CRS)}, crs=UTM_CRS)


# --------------------------------------------------------------------------------------
# _building_density
# --------------------------------------------------------------------------------------


def test_building_density_matches_known_footprint_area(hex_cells, hex_utm_by_h3):
    target_hex = hex_cells[0]
    other_target_hex = hex_cells[2]
    centroid0 = hex_utm_by_h3.loc[target_hex, "geometry"].centroid
    centroid2 = hex_utm_by_h3.loc[other_target_hex, "geometry"].centroid

    # Two known-area footprints, each centered exactly on a different hex's own UTM centroid,
    # so centroid-based assignment is unambiguous. Assignment does not clip to the hex, so a
    # box need not fit entirely inside it for the density formula to hold — only its centroid
    # location matters.
    box0 = box(centroid0.x - 20, centroid0.y - 30, centroid0.x + 20, centroid0.y + 30)  # 40x60=2400 m2
    box2 = box(centroid2.x - 15, centroid2.y - 15, centroid2.x + 15, centroid2.y + 15)  # 30x30=900 m2
    assert box0.area == pytest.approx(2400.0)
    assert box2.area == pytest.approx(900.0)

    buildings_utm = gpd.GeoDataFrame({"geometry": [box0, box2]}, crs=UTM_CRS)

    density = _building_density(buildings_utm, hex_utm_by_h3, RES)

    assert set(density.index) == set(hex_utm_by_h3.index)
    hex_area0 = hex_utm_by_h3.loc[target_hex, "geometry"].area
    hex_area2 = hex_utm_by_h3.loc[other_target_hex, "geometry"].area
    assert density.loc[target_hex] == pytest.approx(2400.0 / hex_area0, rel=1e-6)
    assert density.loc[other_target_hex] == pytest.approx(900.0 / hex_area2, rel=1e-6)

    untouched = [c for c in hex_utm_by_h3.index if c not in (target_hex, other_target_hex)]
    assert (density.loc[untouched] == 0.0).all()


def test_building_density_no_buildings_is_all_zero(hex_utm_by_h3):
    density = _building_density(_empty_utm_gdf(), hex_utm_by_h3, RES)
    assert (density == 0.0).all()
    assert set(density.index) == set(hex_utm_by_h3.index)


def test_building_density_is_clipped_to_one(hex_cells, hex_utm_by_h3):
    target_hex = hex_cells[1]
    hex_geom = hex_utm_by_h3.loc[target_hex, "geometry"]
    centroid = hex_geom.centroid
    hex_area = hex_geom.area

    # A footprint far larger than the hex itself (but centered inside it) must still clip to 1.
    side = (hex_area * 5) ** 0.5
    huge_box = box(
        centroid.x - side / 2, centroid.y - side / 2, centroid.x + side / 2, centroid.y + side / 2
    )
    buildings_utm = gpd.GeoDataFrame({"geometry": [huge_box]}, crs=UTM_CRS)

    density = _building_density(buildings_utm, hex_utm_by_h3, RES)
    assert density.loc[target_hex] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# _road_density
# --------------------------------------------------------------------------------------


def test_road_density_matches_known_length_via_length_column(hex_cells, hex_utm_by_h3):
    target_hex = hex_cells[4]
    centroid = hex_utm_by_h3.loc[target_hex, "geometry"].centroid

    # A straight 200 m line whose midpoint is exactly the hex centroid.
    line = LineString([(centroid.x - 100, centroid.y), (centroid.x + 100, centroid.y)])
    edges_utm = gpd.GeoDataFrame({"geometry": [line], "length": [200.0]}, crs=UTM_CRS)

    density = _road_density(edges_utm, hex_utm_by_h3, RES)

    hex_area_km2 = hex_utm_by_h3.loc[target_hex, "geometry"].area / 1e6
    expected = (200.0 / 1000.0) / hex_area_km2
    assert density.loc[target_hex] == pytest.approx(expected, rel=1e-6)

    untouched = [c for c in hex_utm_by_h3.index if c != target_hex]
    assert (density.loc[untouched] == 0.0).all()


def test_road_density_falls_back_to_geometry_length_without_length_column(hex_cells, hex_utm_by_h3):
    target_hex = hex_cells[3]
    centroid = hex_utm_by_h3.loc[target_hex, "geometry"].centroid

    line = LineString([(centroid.x, centroid.y - 75), (centroid.x, centroid.y + 75)])  # 150 m
    edges_utm = gpd.GeoDataFrame({"geometry": [line]}, crs=UTM_CRS)
    assert "length" not in edges_utm.columns
    assert line.length == pytest.approx(150.0)

    density = _road_density(edges_utm, hex_utm_by_h3, RES)

    hex_area_km2 = hex_utm_by_h3.loc[target_hex, "geometry"].area / 1e6
    expected = (line.length / 1000.0) / hex_area_km2
    assert density.loc[target_hex] == pytest.approx(expected, rel=1e-6)


def test_road_density_no_edges_is_all_zero(hex_utm_by_h3):
    density = _road_density(_empty_utm_gdf(), hex_utm_by_h3, RES)
    assert (density == 0.0).all()


# --------------------------------------------------------------------------------------
# _distance_to_nearest
# --------------------------------------------------------------------------------------


def test_distance_to_nearest_known_offset():
    origin = gpd.GeoSeries([Point(0.0, 0.0)], crs=UTM_CRS)
    square = box(500.0, -25.0, 550.0, 25.0)  # nearest edge is the vertical segment x=500
    polys = gpd.GeoSeries([square], crs=UTM_CRS)

    dist = _distance_to_nearest(origin, polys, cap=10_000.0)

    assert isinstance(dist, np.ndarray)
    assert dist.shape == (1,)
    assert dist[0] == pytest.approx(500.0, abs=1.0)


def test_distance_to_nearest_no_polygons_returns_cap():
    points = gpd.GeoSeries([Point(0.0, 0.0), Point(123.0, 456.0)], crs=UTM_CRS)
    empty_polys = gpd.GeoSeries([], crs=UTM_CRS)

    dist = _distance_to_nearest(points, empty_polys, cap=10_000.0)

    assert dist.shape == (2,)
    assert np.all(dist == 10_000.0)


def test_distance_to_nearest_caps_far_points():
    points = gpd.GeoSeries([Point(0.0, 0.0)], crs=UTM_CRS)
    far_square = box(50_000.0, -25.0, 50_050.0, 25.0)  # much further than the cap
    polys = gpd.GeoSeries([far_square], crs=UTM_CRS)

    dist = _distance_to_nearest(points, polys, cap=10_000.0)

    assert dist[0] == pytest.approx(10_000.0)


def test_distance_to_nearest_vectorized_multiple_points():
    near_pt = Point(0.0, 0.0)
    far_pt = Point(-9_000.0, -9_000.0)
    points = gpd.GeoSeries([near_pt, far_pt], crs=UTM_CRS)
    square = box(500.0, -25.0, 550.0, 25.0)
    polys = gpd.GeoSeries([square], crs=UTM_CRS)

    dist = _distance_to_nearest(points, polys, cap=10_000.0)

    assert dist[0] == pytest.approx(500.0, abs=1.0)
    assert dist[1] == pytest.approx(10_000.0)  # true distance is much larger than the cap


# --------------------------------------------------------------------------------------
# _is_park
# --------------------------------------------------------------------------------------


def test_is_park_inside_true_outside_false():
    park = box(-100.0, -100.0, 100.0, 100.0)
    parks = gpd.GeoSeries([park], crs=UTM_CRS)
    points = gpd.GeoSeries([Point(0.0, 0.0), Point(500.0, 500.0)], crs=UTM_CRS)

    is_park = _is_park(points, parks)

    assert is_park.dtype == bool
    assert is_park.tolist() == [True, False]


def test_is_park_no_parks_all_false():
    points = gpd.GeoSeries([Point(0.0, 0.0), Point(1.0, 1.0)], crs=UTM_CRS)
    empty_parks = gpd.GeoSeries([], crs=UTM_CRS)

    is_park = _is_park(points, empty_parks)

    assert is_park.tolist() == [False, False]


def test_is_park_real_hex_centroids(hex_cells, hex_utm_by_h3):
    # One real hex polygon used directly as the "park"; its own centroid must register True,
    # a different hex's centroid (outside that park) must register False.
    park_hex_id = hex_cells[0]
    other_hex_id = hex_cells[1]
    parks = gpd.GeoSeries([hex_utm_by_h3.loc[park_hex_id, "geometry"]], crs=UTM_CRS)

    centroids = gpd.GeoSeries(
        [
            hex_utm_by_h3.loc[park_hex_id, "geometry"].centroid,
            hex_utm_by_h3.loc[other_hex_id, "geometry"].centroid,
        ],
        crs=UTM_CRS,
    )

    is_park = _is_park(centroids, parks)
    assert is_park.tolist() == [True, False]


def test_is_park_dataframe_geometry_column_input(hex_cells, hex_gdf_utm):
    """`_is_park`/`_distance_to_nearest` are also called with `.geometry` of a GeoDataFrame
    (not a bare GeoSeries) in fetch_osm_features_per_hex — exercise that exact input shape."""
    parks_gdf = gpd.GeoDataFrame(
        {"geometry": [hex_gdf_utm.set_index("h3").loc[hex_cells[0], "geometry"]]}, crs=UTM_CRS
    )
    centroids = hex_gdf_utm.geometry.centroid

    is_park = _is_park(centroids, parks_gdf.geometry)
    assert bool(is_park[0]) is True
