"""Offline tests for heat_island.db — the DuckDB dashboard contract.

No network access; hex data is synthetic (shapely boxes, plausible numeric ranges).
boundary/metrics are duck-typed via types.SimpleNamespace (db.py must not import
boundary.py or model.py).
"""

from __future__ import annotations

import types

import duckdb
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from heat_island import db

N_HEXES = 20

CITIES_COLUMNS = ["city_id", "name", "country", "centroid_lat", "centroid_lon", "processed_at", "n_hexes"]
HEXES_COLUMNS = [
    "city_id",
    "h3",
    "lat",
    "lon",
    "geometry_wkt",
    "mean_lst_c",
    "ndvi",
    "ndbi",
    "ndwi",
    "albedo",
    "elevation",
    "building_density",
    "road_density",
    "dist_water_m",
    "dist_park_m",
    "plantable_fraction",
    "tree_fraction",
    "median_income",
    "pct_over_65",
    "pct_under_5",
    "predicted_lst_c",
    "predicted_cooling_c",
    "cooling_uncertainty_c",
    "priority_score",
]
MODEL_METRICS_COLUMNS = ["city_id", "r2", "mae", "n_train", "trained_at"]
FEATURE_IMPORTANCE_COLUMNS = ["city_id", "feature", "mean_abs_shap"]

SHAP_IMPORTANCE = {
    "ndvi": 1.2345,
    "ndbi": 0.8765,
    "ndwi": 0.4123,
    "albedo": 0.2019,
    "elevation": 0.1502,
    "building_density": 0.9004,
    "road_density": 0.3301,
    "dist_water_m": 0.0752,
    "dist_park_m": 0.0601,
}


# --------------------------------------------------------------------------------------
# Synthetic data helpers
# --------------------------------------------------------------------------------------


def make_hex_df(
    n: int = N_HEXES,
    seed: int = 0,
    id_prefix: str = "h3",
    city_lat: float = 51.0543,
    city_lon: float = 3.7174,
    with_demographics: bool = True,
    with_plantability: bool = True,
) -> pd.DataFrame:
    """~n synthetic hex rows: unique h3-like strings, shapely box geometries, plausible values.

    Demographics present for half the rows (even i) and NaN for the other half (odd i) when
    with_demographics=True; columns omitted entirely when False.

    plantable_fraction/tree_fraction/cooling_uncertainty_c present with plausible values when
    with_plantability=True (default), but NaN on a subset of rows (i % 4 == 0 for the land-cover
    pair, i % 3 == 0 for the uncertainty column -- deliberately different, overlapping-but-not-
    identical patterns from the demographics i % 2 one) to prove the per-row NULL path
    independently of both the whole-column-missing case and the demographics NaN pattern;
    columns omitted entirely when with_plantability=False (exercises the "whole column absent ->
    NULL" duck-typed leniency, same as demographics).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lat = city_lat + i * 0.001
        lon = city_lon + i * 0.001
        geom = box(lon - 0.0005, lat - 0.0005, lon + 0.0005, lat + 0.0005)
        row: dict[str, object] = dict(
            h3=f"{id_prefix}-{i:04d}",
            lat=lat,
            lon=lon,
            geometry=geom,
            mean_lst_c=25.0 + rng.uniform(0, 15),
            ndvi=rng.uniform(-0.1, 0.9),
            ndbi=rng.uniform(-0.3, 0.3),
            ndwi=rng.uniform(-0.3, 0.3),
            albedo=rng.uniform(0.05, 0.3),
            elevation=rng.uniform(0, 50),
            building_density=rng.uniform(0, 1),
            road_density=rng.uniform(0, 20),
            dist_water_m=rng.uniform(0, 10000),
            dist_park_m=rng.uniform(0, 10000),
            predicted_lst_c=25.0 + rng.uniform(0, 15),
            predicted_cooling_c=rng.uniform(0, 3),
            priority_score=rng.uniform(0, 1),
        )
        if with_plantability:
            row["plantable_fraction"] = np.nan if i % 4 == 0 else float(rng.uniform(0.0, 1.0))
            row["tree_fraction"] = np.nan if i % 4 == 0 else float(rng.uniform(0.0, 1.0))
            row["cooling_uncertainty_c"] = np.nan if i % 3 == 0 else float(rng.uniform(0.0, 1.5))
        if with_demographics:
            if i % 2 == 0:
                row["median_income"] = float(rng.uniform(20000, 95000))
                row["pct_over_65"] = float(rng.uniform(5, 30))
                row["pct_under_5"] = float(rng.uniform(2, 10))
            else:
                row["median_income"] = np.nan
                row["pct_over_65"] = np.nan
                row["pct_under_5"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_boundary(
    city_id: str = "ghent-belgie",
    name: str = "Gent",
    country: str = "Belgium",
    centroid_lat: float = 51.0543,
    centroid_lon: float = 3.7174,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        city_id=city_id,
        name=name,
        country=country,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
    )


def make_metrics(
    r2: float = 0.7234,
    mae: float = 1.85,
    n_train: int = N_HEXES,
    shap_importance: dict[str, float] | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        r2=r2,
        mae=mae,
        n_train=n_train,
        shap_importance=dict(SHAP_IMPORTANCE) if shap_importance is None else shap_importance,
    )


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [r[0] for r in rows]


def _count(con: duckdb.DuckDBPyConnection, table: str, city_id: str | None = None) -> int:
    if city_id is None:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return con.execute(f"SELECT COUNT(*) FROM {table} WHERE city_id = ?", [city_id]).fetchone()[0]


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def test_schema_contract(tmp_path):
    """The dashboard contract test: exact column name lists, in order, for all 4 tables."""
    db_path = tmp_path / "heat.duckdb"
    con = db.connect(db_path)
    try:
        assert _columns(con, "cities") == CITIES_COLUMNS
        assert _columns(con, "hexes") == HEXES_COLUMNS
        assert len(HEXES_COLUMNS) == 24
        assert _columns(con, "model_metrics") == MODEL_METRICS_COLUMNS
        assert _columns(con, "feature_importance") == FEATURE_IMPORTANCE_COLUMNS
    finally:
        con.close()


def test_connect_creates_parent_dirs(tmp_path):
    db_path = tmp_path / "nested" / "sub" / "heat.duckdb"
    assert not db_path.parent.exists()
    con = db.connect(db_path)
    con.close()
    assert db_path.exists()


def test_upsert_roundtrip(tmp_path):
    db_path = tmp_path / "heat.duckdb"
    hex_df = make_hex_df()
    boundary = make_boundary()
    metrics = make_metrics()

    db.upsert_city(db_path, boundary, hex_df, metrics)

    con = duckdb.connect(str(db_path))
    try:
        assert _count(con, "cities") == 1
        assert _count(con, "hexes") == N_HEXES
        assert _count(con, "model_metrics") == 1
        assert _count(con, "feature_importance") == len(metrics.shap_importance)

        city_row = con.execute(
            "SELECT city_id, name, country, n_hexes FROM cities WHERE city_id = ?", [boundary.city_id]
        ).fetchone()
        assert city_row == (boundary.city_id, boundary.name, boundary.country, N_HEXES)

        # spot check a known h3's ndvi, rounded to 4dp
        h3_id = "h3-0000"
        expected_ndvi = round(float(hex_df.loc[hex_df["h3"] == h3_id, "ndvi"].iloc[0]), 4)
        got_ndvi = con.execute("SELECT ndvi FROM hexes WHERE h3 = ?", [h3_id]).fetchone()[0]
        assert got_ndvi == pytest.approx(expected_ndvi)

        # spot check lat/lon rounding (6dp) and geometry_wkt population
        got_lat, got_lon, got_wkt = con.execute(
            "SELECT lat, lon, geometry_wkt FROM hexes WHERE h3 = ?", [h3_id]
        ).fetchone()
        expected_lat = round(float(hex_df.loc[hex_df["h3"] == h3_id, "lat"].iloc[0]), 6)
        assert got_lat == pytest.approx(expected_lat)
        assert got_wkt.startswith("POLYGON")

        # demographics: even rows have values, odd rows are NULL (NaN -> NULL)
        even_income = con.execute("SELECT median_income FROM hexes WHERE h3 = ?", ["h3-0000"]).fetchone()[0]
        assert even_income is not None
        odd_income = con.execute("SELECT median_income FROM hexes WHERE h3 = ?", ["h3-0001"]).fetchone()[0]
        assert odd_income is None
        odd_pct65 = con.execute("SELECT pct_over_65 FROM hexes WHERE h3 = ?", ["h3-0001"]).fetchone()[0]
        assert odd_pct65 is None

        # spot check plantable_fraction (4dp) / cooling_uncertainty_c (2dp) rounding on a row
        # where the synthetic helper's NaN pattern (i % 4 == 0, i % 3 == 0) doesn't apply
        # (h3-0001 -> i=1)
        expected_plantable = round(
            float(hex_df.loc[hex_df["h3"] == "h3-0001", "plantable_fraction"].iloc[0]), 4
        )
        expected_uncertainty = round(
            float(hex_df.loc[hex_df["h3"] == "h3-0001", "cooling_uncertainty_c"].iloc[0]), 2
        )
        got_plantable, got_uncertainty = con.execute(
            "SELECT plantable_fraction, cooling_uncertainty_c FROM hexes WHERE h3 = ?", ["h3-0001"]
        ).fetchone()
        assert got_plantable == pytest.approx(expected_plantable)
        assert got_uncertainty == pytest.approx(expected_uncertainty)

        # per-row NaN -> NULL: h3-0000 (i=0) hits both the i%4==0 and i%3==0 NaN patterns, so
        # plantable_fraction/tree_fraction/cooling_uncertainty_c are all NULL for that one row
        # even though the columns themselves are present (unlike the whole-column-absent case
        # covered separately below).
        null_plantable, null_tree, null_uncertainty = con.execute(
            "SELECT plantable_fraction, tree_fraction, cooling_uncertainty_c FROM hexes WHERE h3 = ?",
            ["h3-0000"],
        ).fetchone()
        assert null_plantable is None
        assert null_tree is None
        assert null_uncertainty is None

        # model metrics + feature importance content
        r2, mae, n_train = con.execute(
            "SELECT r2, mae, n_train FROM model_metrics WHERE city_id = ?", [boundary.city_id]
        ).fetchone()
        assert r2 == pytest.approx(metrics.r2)
        assert mae == pytest.approx(metrics.mae)
        assert n_train == metrics.n_train

        shap_rows = dict(
            con.execute(
                "SELECT feature, mean_abs_shap FROM feature_importance WHERE city_id = ?", [boundary.city_id]
            ).fetchall()
        )
        assert shap_rows.keys() == metrics.shap_importance.keys()
        for feature, value in metrics.shap_importance.items():
            assert shap_rows[feature] == pytest.approx(value)
    finally:
        con.close()


def test_upsert_idempotent(tmp_path):
    db_path = tmp_path / "heat.duckdb"
    hex_df = make_hex_df()
    boundary = make_boundary()
    metrics = make_metrics()

    db.upsert_city(db_path, boundary, hex_df, metrics)
    db.upsert_city(db_path, boundary, hex_df, metrics)  # re-run must not duplicate

    con = duckdb.connect(str(db_path))
    try:
        assert _count(con, "cities") == 1
        assert _count(con, "hexes") == N_HEXES
        assert _count(con, "model_metrics") == 1
        assert _count(con, "feature_importance") == len(metrics.shap_importance)
    finally:
        con.close()


def test_upsert_replaces_changed_values(tmp_path):
    """A second upsert with different data should fully replace, not merge, the old rows."""
    db_path = tmp_path / "heat.duckdb"
    boundary = make_boundary()

    db.upsert_city(db_path, boundary, make_hex_df(seed=1), make_metrics(r2=0.5))
    db.upsert_city(db_path, boundary, make_hex_df(seed=2), make_metrics(r2=0.9))

    con = duckdb.connect(str(db_path))
    try:
        assert _count(con, "cities") == 1
        assert _count(con, "hexes") == N_HEXES
        r2 = con.execute("SELECT r2 FROM model_metrics WHERE city_id = ?", [boundary.city_id]).fetchone()[0]
        assert r2 == pytest.approx(0.9)
    finally:
        con.close()


def test_remove_and_list(tmp_path):
    db_path = tmp_path / "heat.duckdb"
    b1 = make_boundary(city_id="ghent-belgie", name="Gent", country="Belgium")
    b2 = make_boundary(
        city_id="porto-portugal", name="Porto", country="Portugal", centroid_lat=41.1579, centroid_lon=-8.6291
    )
    hex_df1 = make_hex_df(id_prefix="g", city_lat=51.0543, city_lon=3.7174)
    hex_df2 = make_hex_df(id_prefix="p", city_lat=41.1579, city_lon=-8.6291)
    m1 = make_metrics(r2=0.65)
    m2 = make_metrics(r2=0.80)

    db.upsert_city(db_path, b1, hex_df1, m1)
    db.upsert_city(db_path, b2, hex_df2, m2)

    removed = db.remove_city(db_path, b1.city_id)
    assert removed == N_HEXES

    con = duckdb.connect(str(db_path))
    try:
        # city 1 fully gone from all four tables
        assert _count(con, "cities", b1.city_id) == 0
        assert _count(con, "hexes", b1.city_id) == 0
        assert _count(con, "model_metrics", b1.city_id) == 0
        assert _count(con, "feature_importance", b1.city_id) == 0

        # city 2 left intact
        assert _count(con, "cities", b2.city_id) == 1
        assert _count(con, "hexes", b2.city_id) == N_HEXES
        assert _count(con, "model_metrics", b2.city_id) == 1
        assert _count(con, "feature_importance", b2.city_id) == len(m2.shap_importance)
    finally:
        con.close()

    # removing an absent / already-removed city returns 0, doesn't raise
    assert db.remove_city(db_path, b1.city_id) == 0
    assert db.remove_city(db_path, "no-such-city") == 0

    cities = db.list_cities(db_path)
    assert list(cities.columns) == CITIES_COLUMNS + ["r2", "mae", "n_train", "trained_at"]
    assert list(cities["city_id"]) == [b2.city_id]
    assert cities.loc[cities["city_id"] == b2.city_id, "r2"].iloc[0] == pytest.approx(m2.r2)


def test_list_cities_ordered_by_name_and_missing_db(tmp_path):
    # missing db file: list_cities must not raise, and must return the right (empty) shape
    db_path = tmp_path / "sub" / "heat.duckdb"
    assert not db_path.exists()
    empty = db.list_cities(db_path)
    assert list(empty.columns) == CITIES_COLUMNS + ["r2", "mae", "n_train", "trained_at"]
    assert len(empty) == 0
    assert db_path.exists()  # connect()'s CREATE IF NOT EXISTS created the file

    # order by name (not city_id / insertion order)
    b_zebra = make_boundary(city_id="zebra-city", name="Zebraville")
    b_apple = make_boundary(city_id="apple-city", name="Appleton")
    db.upsert_city(db_path, b_zebra, make_hex_df(id_prefix="z"), make_metrics())
    db.upsert_city(db_path, b_apple, make_hex_df(id_prefix="a"), make_metrics())

    cities = db.list_cities(db_path)
    assert list(cities["name"]) == ["Appleton", "Zebraville"]


def test_missing_demographic_columns(tmp_path):
    """hex_df WITHOUT the demographic columns at all -> upsert works, NULLs stored."""
    db_path = tmp_path / "heat.duckdb"
    hex_df = make_hex_df(with_demographics=False)
    for col in ("median_income", "pct_over_65", "pct_under_5"):
        assert col not in hex_df.columns
    boundary = make_boundary()
    metrics = make_metrics()

    db.upsert_city(db_path, boundary, hex_df, metrics)

    con = duckdb.connect(str(db_path))
    try:
        assert _count(con, "hexes") == N_HEXES
        n_null = con.execute(
            "SELECT COUNT(*) FROM hexes "
            "WHERE median_income IS NULL AND pct_over_65 IS NULL AND pct_under_5 IS NULL"
        ).fetchone()[0]
        assert n_null == N_HEXES
    finally:
        con.close()


def test_missing_plantability_and_uncertainty_columns(tmp_path):
    """hex_df WITHOUT plantable_fraction/tree_fraction/cooling_uncertainty_c -> NULLs stored.

    Same duck-typed leniency as demographics (ARCHITECTURE.md): a caller that hasn't run the
    land-cover fetch or doesn't have fold models around yet can still upsert.
    """
    db_path = tmp_path / "heat.duckdb"
    hex_df = make_hex_df(with_plantability=False)
    for col in ("plantable_fraction", "tree_fraction", "cooling_uncertainty_c"):
        assert col not in hex_df.columns
    boundary = make_boundary()
    metrics = make_metrics()

    db.upsert_city(db_path, boundary, hex_df, metrics)

    con = duckdb.connect(str(db_path))
    try:
        assert _count(con, "hexes") == N_HEXES
        n_null = con.execute(
            "SELECT COUNT(*) FROM hexes "
            "WHERE plantable_fraction IS NULL AND tree_fraction IS NULL AND cooling_uncertainty_c IS NULL"
        ).fetchone()[0]
        assert n_null == N_HEXES
    finally:
        con.close()


def test_upsert_accepts_geometry_wkt_column(tmp_path):
    """hex_df MAY provide geometry_wkt directly instead of a shapely geometry column."""
    db_path = tmp_path / "heat.duckdb"
    hex_df = make_hex_df()
    hex_df["geometry_wkt"] = hex_df["geometry"].apply(lambda g: g.wkt)
    hex_df = hex_df.drop(columns=["geometry"])
    boundary = make_boundary()
    metrics = make_metrics()

    db.upsert_city(db_path, boundary, hex_df, metrics)

    con = duckdb.connect(str(db_path))
    try:
        wkt = con.execute("SELECT geometry_wkt FROM hexes WHERE h3 = ?", ["h3-0000"]).fetchone()[0]
        assert wkt.startswith("POLYGON")
    finally:
        con.close()


def test_upsert_missing_geometry_raises(tmp_path):
    db_path = tmp_path / "heat.duckdb"
    hex_df = make_hex_df().drop(columns=["geometry"])
    with pytest.raises(Exception):
        db.upsert_city(db_path, make_boundary(), hex_df, make_metrics())


def test_migrates_old_21_column_hexes_table(tmp_path):
    """A heat.duckdb written by v0.1 (21-col hexes) must gain the v0.2 columns on connect
    and accept a fresh upsert; pre-migration rows read as NULL for the new columns."""
    db_path = tmp_path / "heat.duckdb"
    old_cols = [c for c in db.HEX_COLUMNS
                if c not in ("plantable_fraction", "tree_fraction", "cooling_uncertainty_c")]
    con = duckdb.connect(str(db_path))
    decls = ", ".join(f'"{c}" TEXT' if c in ("city_id", "h3", "geometry_wkt") else f'"{c}" DOUBLE'
                      for c in old_cols)
    con.execute(f"CREATE TABLE hexes ({decls}, PRIMARY KEY (city_id, h3))")
    con.execute(
        "INSERT INTO hexes (city_id, h3, lat, lon, geometry_wkt, mean_lst_c) "
        "VALUES ('old-city', 'aaa', 1.0, 2.0, 'POLYGON EMPTY', 33.3)"
    )
    con.close()

    con = db.connect(db_path)
    cols = _columns(con, "hexes")
    assert set(db.HEX_COLUMNS) <= set(cols)
    row = con.execute(
        "SELECT plantable_fraction, tree_fraction, cooling_uncertainty_c "
        "FROM hexes WHERE city_id = 'old-city'"
    ).fetchone()
    assert all(v is None for v in row)
    con.close()

    db.upsert_city(db_path, make_boundary(), make_hex_df(), make_metrics())
    con = duckdb.connect(str(db_path))
    assert _count(con, "hexes", "old-city") == 1  # untouched
    con.close()
