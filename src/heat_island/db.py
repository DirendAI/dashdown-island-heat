"""DuckDB persistence layer.

This is the dashboard's read contract: the schema below is EXACT (see ARCHITECTURE.md,
"db.py (dashboard contract — EXACT schema)") and must not drift without updating that doc.

Decoupling note: this module intentionally does NOT import boundary.py or model.py. The
``boundary`` and ``metrics`` arguments below are duck-typed:
  - ``boundary``: any object with attributes ``city_id``, ``name``, ``country``,
    ``centroid_lat``, ``centroid_lon`` (matches ``boundary.CityBoundary``).
  - ``metrics``: any object with attributes ``r2``, ``mae``, ``n_train``,
    ``shap_importance`` (``dict[str, float]``) (matches ``model.ModelResult``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

import duckdb
import numpy as np
import pandas as pd

from .util import PipelineError, get_logger

log = get_logger(__name__)


class BoundaryLike(Protocol):
    """Structural type for the ``boundary`` argument — see module docstring."""

    city_id: str
    name: str
    country: str
    centroid_lat: float
    centroid_lon: float


class MetricsLike(Protocol):
    """Structural type for the ``metrics`` argument — see module docstring."""

    r2: float
    mae: float
    n_train: int
    shap_importance: Mapping[str, float]


# --- schema (EXACT — copied verbatim from ARCHITECTURE.md) -----------------------------

_CREATE_CITIES_SQL = """
CREATE TABLE IF NOT EXISTS cities(
  city_id TEXT PRIMARY KEY, name TEXT NOT NULL, country TEXT,
  centroid_lat DOUBLE, centroid_lon DOUBLE, processed_at TIMESTAMP, n_hexes INTEGER);
"""

_CREATE_HEXES_SQL = """
CREATE TABLE IF NOT EXISTS hexes(
  city_id TEXT, h3 TEXT, lat DOUBLE, lon DOUBLE, geometry_wkt TEXT,
  mean_lst_c DOUBLE, ndvi DOUBLE, ndbi DOUBLE, ndwi DOUBLE, albedo DOUBLE,
  elevation DOUBLE, building_density DOUBLE, road_density DOUBLE,
  dist_water_m DOUBLE, dist_park_m DOUBLE,
  median_income DOUBLE, pct_over_65 DOUBLE, pct_under_5 DOUBLE,
  predicted_lst_c DOUBLE, predicted_cooling_c DOUBLE, priority_score DOUBLE,
  PRIMARY KEY (city_id, h3));
"""

_CREATE_MODEL_METRICS_SQL = """
CREATE TABLE IF NOT EXISTS model_metrics(
  city_id TEXT PRIMARY KEY, r2 DOUBLE, mae DOUBLE, n_train INTEGER, trained_at TIMESTAMP);
"""

_CREATE_FEATURE_IMPORTANCE_SQL = """
CREATE TABLE IF NOT EXISTS feature_importance(
  city_id TEXT, feature TEXT, mean_abs_shap DOUBLE, PRIMARY KEY (city_id, feature));
"""

# Exact hexes-table column order (21 columns) — also the column order used for the
# DataFrame we register with duckdb before INSERT INTO hexes SELECT ... FROM df.
HEX_COLUMNS: list[str] = [
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
    "median_income",
    "pct_over_65",
    "pct_under_5",
    "predicted_lst_c",
    "predicted_cooling_c",
    "priority_score",
]

# Rounding policy for the numeric hexes columns ("round floats sensibly" per ARCHITECTURE.md).
_ROUND_4DP = ("ndvi", "ndbi", "ndwi", "albedo", "priority_score")
_ROUND_2DP = (
    "mean_lst_c",
    "elevation",
    "building_density",
    "road_density",
    "dist_water_m",
    "dist_park_m",
    "predicted_lst_c",
    "predicted_cooling_c",
    # pct_over_65 / pct_under_5 aren't called out explicitly in the contract; treated as
    # 2dp like the other percent/measurement columns rather than left unrounded.
    "pct_over_65",
    "pct_under_5",
)
_ROUND_0DP = ("median_income",)


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open (creating parent dirs and the schema if needed) the heat-island DuckDB file."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_CREATE_CITIES_SQL)
    con.execute(_CREATE_HEXES_SQL)
    con.execute(_CREATE_MODEL_METRICS_SQL)
    con.execute(_CREATE_FEATURE_IMPORTANCE_SQL)
    return con


def _geometry_to_wkt(value: Any) -> str | None:
    """Duck-typed geometry -> WKT string. Accepts shapely geometries, WKT strings, or None/NaN."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    wkt = getattr(value, "wkt", None)
    if wkt is not None:
        return wkt
    return str(value)


def _numeric_column(hex_df: pd.DataFrame, col: str, decimals: int) -> pd.Series:
    if col in hex_df.columns:
        return pd.to_numeric(hex_df[col], errors="coerce").round(decimals)
    return pd.Series(np.nan, index=hex_df.index)


def _build_hex_frame(city_id: str, hex_df: pd.DataFrame) -> pd.DataFrame:
    """Build a plain pandas frame with exactly the hexes-table columns, in schema order."""
    missing_required = [c for c in ("h3", "lat", "lon") if c not in hex_df.columns]
    if missing_required:
        raise PipelineError(f"hex_df is missing required column(s): {missing_required}")

    out = pd.DataFrame(index=hex_df.index)
    out["city_id"] = city_id
    out["h3"] = hex_df["h3"].astype(str)
    out["lat"] = _numeric_column(hex_df, "lat", 6)
    out["lon"] = _numeric_column(hex_df, "lon", 6)

    if "geometry_wkt" in hex_df.columns:
        out["geometry_wkt"] = hex_df["geometry_wkt"].map(_geometry_to_wkt)
    elif "geometry" in hex_df.columns:
        out["geometry_wkt"] = hex_df["geometry"].map(_geometry_to_wkt)
    else:
        raise PipelineError("hex_df must contain a 'geometry' or 'geometry_wkt' column")

    for col in _ROUND_4DP:
        out[col] = _numeric_column(hex_df, col, 4)
    for col in _ROUND_2DP:
        out[col] = _numeric_column(hex_df, col, 2)
    for col in _ROUND_0DP:
        out[col] = _numeric_column(hex_df, col, 0)

    return out[HEX_COLUMNS].reset_index(drop=True)


def upsert_city(db_path: Path, boundary: BoundaryLike, hex_df: pd.DataFrame, metrics: MetricsLike) -> None:
    """Replace a city's rows across all four tables in a single transaction.

    DELETE FROM each table WHERE city_id = ? first (so this is idempotent / re-runnable —
    milestone gate 5: re-running add-city must not duplicate rows), then INSERT fresh rows.
    """
    city_id = boundary.city_id
    frame = _build_hex_frame(city_id, hex_df)
    n_hexes = len(frame)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    log.info("upsert_city: city_id=%s n_hexes=%d", city_id, n_hexes)

    con = connect(db_path)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute("DELETE FROM hexes WHERE city_id = ?", [city_id])
        con.execute("DELETE FROM feature_importance WHERE city_id = ?", [city_id])
        con.execute("DELETE FROM model_metrics WHERE city_id = ?", [city_id])
        con.execute("DELETE FROM cities WHERE city_id = ?", [city_id])

        con.execute(
            "INSERT INTO cities (city_id, name, country, centroid_lat, centroid_lon, "
            "processed_at, n_hexes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                city_id,
                boundary.name,
                boundary.country,
                float(boundary.centroid_lat),
                float(boundary.centroid_lon),
                now,
                n_hexes,
            ],
        )

        con.register("df", frame)
        try:
            columns_sql = ", ".join(HEX_COLUMNS)
            con.execute(f"INSERT INTO hexes ({columns_sql}) SELECT {columns_sql} FROM df")
        finally:
            con.unregister("df")

        con.execute(
            "INSERT INTO model_metrics (city_id, r2, mae, n_train, trained_at) VALUES (?, ?, ?, ?, ?)",
            [city_id, float(metrics.r2), float(metrics.mae), int(metrics.n_train), now],
        )

        shap_importance = metrics.shap_importance or {}
        if shap_importance:
            con.executemany(
                "INSERT INTO feature_importance (city_id, feature, mean_abs_shap) VALUES (?, ?, ?)",
                [(city_id, str(feature), float(value)) for feature, value in shap_importance.items()],
            )

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    log.info("upsert_city: committed city_id=%s (%d hexes, %d shap rows)", city_id, n_hexes, len(shap_importance))


def list_cities(db_path: Path) -> pd.DataFrame:
    """All cities, left-joined with their latest model metrics, ordered by name.

    If the DB file doesn't exist yet, ``connect`` creates it (schema only, no rows), so
    this simply returns an empty frame with the correct columns.
    """
    con = connect(db_path)
    try:
        return con.execute(
            """
            SELECT
                c.city_id, c.name, c.country, c.centroid_lat, c.centroid_lon,
                c.processed_at, c.n_hexes,
                m.r2, m.mae, m.n_train, m.trained_at
            FROM cities c
            LEFT JOIN model_metrics m ON c.city_id = m.city_id
            ORDER BY c.name
            """
        ).df()
    finally:
        con.close()


def remove_city(db_path: Path, city_id: str) -> int:
    """Delete a city's rows from all four tables. Returns the number of hex rows removed."""
    con = connect(db_path)
    try:
        con.execute("BEGIN TRANSACTION")
        n_hexes = con.execute("SELECT COUNT(*) FROM hexes WHERE city_id = ?", [city_id]).fetchone()[0]
        con.execute("DELETE FROM hexes WHERE city_id = ?", [city_id])
        con.execute("DELETE FROM feature_importance WHERE city_id = ?", [city_id])
        con.execute("DELETE FROM model_metrics WHERE city_id = ?", [city_id])
        con.execute("DELETE FROM cities WHERE city_id = ?", [city_id])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    log.info("remove_city: city_id=%s removed %d hex rows", city_id, n_hexes)
    return int(n_hexes)
