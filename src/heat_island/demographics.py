"""US Census demographics per hex — optional enrichment that must never crash the pipeline.

Two keyless Census Bureau HTTP services are combined:
  - TIGERweb (ArcGIS REST `MapServer`) for current census tract polygons intersecting the city.
  - ACS 5-year detail tables (median household income + sex-by-age population counts) for
    every (state, county) pair the intersecting tracts fall in.
Tract-level numbers are then areally interpolated onto the H3 grid (intersection-area weights,
computed in the city's local UTM CRS) to produce one row per hex.

The HTTP layer is injected (`fetch_json`) so this whole module is unit-testable with zero
network access — tests pass a fake that returns canned JSON/GeoJSON. Gated on the boundary's
country being the United States; any failure past that gate (TIGERweb down, ACS down, bad
geometry, ...) is caught, logged at WARNING, and turned into an empty frame — demographics are
a bonus feature, never a pipeline blocker (ARCHITECTURE.md).
"""

from __future__ import annotations

import os
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from .boundary import CityBoundary
from .config import DEMOGRAPHIC_COLS, PipelineConfig
from .util import DataUnavailableError, get_logger, retry_call, utm_crs_for

log = get_logger(__name__)

TIGERWEB_MAPSERVER = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer"
ACS_VINTAGES: tuple[int, ...] = (2023, 2022)

INCOME_VAR = "B19013_001E"
TOTAL_VAR = "B01001_001E"
UNDER5_VARS: tuple[str, ...] = ("B01001_003E", "B01001_027E")
OVER65_VARS: tuple[str, ...] = tuple(f"B01001_{i:03d}E" for i in range(20, 26)) + tuple(
    f"B01001_{i:03d}E" for i in range(44, 50)
)
ACS_VARS: tuple[str, ...] = (INCOME_VAR, TOTAL_VAR, *UNDER5_VARS, *OVER65_VARS)

_INCOME_SENTINEL = -666_666  # Census "not applicable"/error codes are <= this (e.g. -666666666)
_MAX_TIGERWEB_PAGES = 50  # safety cap against a pathological/looping paginated response

FetchJson = Callable[[str, dict[str, Any]], Any]


def _fetch_json(url: str, params: dict[str, Any]) -> Any:
    """Default HTTP transport: GET url?params as JSON, retried via `util.retry_call`."""

    def _get() -> Any:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    return retry_call(_get, what=f"GET {url}")


def _empty() -> pd.DataFrame:
    """Zero-row frame with the contract columns (right dtypes for a clean downstream merge)."""
    data: dict[str, pd.Series] = {"h3": pd.Series(dtype="object")}
    data.update({col: pd.Series(dtype="float64") for col in DEMOGRAPHIC_COLS})
    return pd.DataFrame(data)


def _is_us(boundary: CityBoundary) -> bool:
    return "united states" in boundary.country.lower()


def _acs_url(vintage: int) -> str:
    return f"https://api.census.gov/data/{vintage}/acs/acs5"


def _find_tract_layer_id(fetch_json: FetchJson) -> int:
    """First TIGERweb layer id whose name contains "Census Tracts" (there may be several)."""
    data = fetch_json(TIGERWEB_MAPSERVER, {"f": "json"})
    for layer in data.get("layers", []):
        if "Census Tracts" in str(layer.get("name", "")):
            return int(layer["id"])
    raise DataUnavailableError("TIGERweb MapServer has no layer named like 'Census Tracts'")


def _fetch_tracts(
    fetch_json: FetchJson, layer_id: int, bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """All census tract polygons intersecting `bounds` (EPSG:4326 envelope), GEOID + geometry.

    Paginates via `resultOffset` (advanced by however many features the last page actually
    returned) until the server stops reporting `exceededTransferLimit`.
    """
    minx, miny, maxx, maxy = bounds
    query_url = f"{TIGERWEB_MAPSERVER}/{layer_id}/query"
    features: list[dict] = []
    offset = 0
    for _ in range(_MAX_TIGERWEB_PAGES):
        params = {
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "GEOID",
            "returnGeometry": "true",
            "f": "geojson",
            "resultOffset": offset,
        }
        page = fetch_json(query_url, params)
        feats = page.get("features") or []
        features.extend(feats)
        props = page.get("properties") or {}
        exceeded = bool(page.get("exceededTransferLimit") or props.get("exceededTransferLimit"))
        if not exceeded or not feats:
            break
        offset += len(feats)
    else:
        log.warning(
            "TIGERweb tract pagination hit the %d-page safety cap — results may be incomplete",
            _MAX_TIGERWEB_PAGES,
        )

    if not features:
        return gpd.GeoDataFrame(
            {"GEOID": pd.Series(dtype="object")}, geometry=gpd.GeoSeries([], crs="EPSG:4326")
        )

    tracts = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    return tracts[["GEOID", "geometry"]]


def _sentinel_income(value: float) -> float:
    """Census sentinel/error codes (e.g. -666666666) and non-positive incomes -> NaN."""
    if value <= 0 or value <= _INCOME_SENTINEL:
        return float("nan")
    return value


def _num(value) -> float:
    """Null-safe ACS cell parse: JSON null / missing / unparseable -> NaN, never an exception.

    ACS suppresses some tract cells as null; one bad cell must only lose that tract's value,
    not (via an exception) the entire city's demographics.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_acs_rows(rows: list[list[str]]) -> pd.DataFrame:
    """ACS "array of arrays" response (header row + data rows) -> one row per tract."""
    header, *data_rows = rows
    records = [dict(zip(header, row)) for row in data_rows]

    parsed = []
    for rec in records:
        income = _sentinel_income(_num(rec.get(INCOME_VAR)))
        total = _num(rec.get(TOTAL_VAR))
        under5 = sum(_num(rec.get(v)) for v in UNDER5_VARS)
        over65 = sum(_num(rec.get(v)) for v in OVER65_VARS)
        total_valid = total if total > 0 else float("nan")
        parsed.append(
            {
                "GEOID": f"{rec['state']}{rec['county']}{rec['tract']}",
                "median_income": income,
                "pct_over_65": 100.0 * over65 / total_valid,
                "pct_under_5": 100.0 * under5 / total_valid,
            }
        )
    return pd.DataFrame(parsed, columns=["GEOID", "median_income", "pct_over_65", "pct_under_5"])


def _query_acs(fetch_json: FetchJson, state: str, county: str) -> list[list[str]] | None:
    """Try ACS_VINTAGES in order; return the first response with data rows, else None."""
    api_key = os.environ.get("CENSUS_API_KEY")
    for vintage in ACS_VINTAGES:
        params = {
            "get": ",".join(ACS_VARS),
            "for": "tract:*",
            "in": f"state:{state} county:{county}",
        }
        if api_key:
            params["key"] = api_key
        try:
            rows = fetch_json(_acs_url(vintage), params)
        except Exception as exc:
            log.warning(
                "ACS %d query failed for state=%s county=%s: %s — trying next vintage",
                vintage, state, county, exc,
            )
            continue
        if isinstance(rows, list) and len(rows) > 1:
            return rows
        log.warning(
            "ACS %d returned no data rows for state=%s county=%s — trying next vintage",
            vintage, state, county,
        )
    return None


def _tract_metrics(fetch_json: FetchJson, tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    """ACS metrics per tract GEOID — one request per unique (state, county) in `tracts`."""
    pairs = sorted({(geoid[0:2], geoid[2:5]) for geoid in tracts["GEOID"]})
    frames = []
    for state, county in pairs:
        rows = _query_acs(fetch_json, state, county)
        if rows is None:
            log.warning("no ACS data in any vintage %s for state=%s county=%s", ACS_VINTAGES, state, county)
            continue
        frames.append(_parse_acs_rows(rows))

    if not frames:
        return pd.DataFrame(columns=["GEOID", "median_income", "pct_over_65", "pct_under_5"])
    return pd.concat(frames, ignore_index=True)


def _areal_interpolate(
    hex_gdf: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame, metrics: pd.DataFrame, utm_crs
) -> pd.DataFrame:
    """Per-hex intersection-area-weighted mean of each tract metric.

    `gpd.overlay(hexes, tracts, how="intersection")` (UTM) gives every (hex, tract) piece with
    its own geometry; weight = that piece's area. For each metric, hexes get
    `sum(w * val where val notna) / sum(w where val notna)` — tracts with a NaN metric (no ACS
    data, or a sentinel-scrubbed income) simply contribute no weight to that metric, and a hex
    touching only such tracts ends up NaN for it.
    """
    if len(tracts) == 0 or len(metrics) == 0:
        return pd.DataFrame(columns=["h3", *DEMOGRAPHIC_COLS])

    hexes_utm = hex_gdf[["h3", "geometry"]].to_crs(utm_crs)
    tracts_utm = tracts[["GEOID", "geometry"]].to_crs(utm_crs)
    overlay = gpd.overlay(hexes_utm, tracts_utm, how="intersection", keep_geom_type=True)
    if len(overlay) == 0:
        return pd.DataFrame(columns=["h3", *DEMOGRAPHIC_COLS])

    overlay = overlay.merge(metrics, on="GEOID", how="left")
    weight = overlay.geometry.area.to_numpy()

    frame = pd.DataFrame({"h3": overlay["h3"].to_numpy()})
    for col in DEMOGRAPHIC_COLS:
        vals = overlay[col].to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        frame[f"num_{col}"] = np.where(valid, weight * vals, 0.0)
        frame[f"den_{col}"] = np.where(valid, weight, 0.0)

    grouped = frame.groupby("h3", as_index=False).sum(numeric_only=True)
    result = pd.DataFrame({"h3": grouped["h3"]})
    for col in DEMOGRAPHIC_COLS:
        denom = grouped[f"den_{col}"].replace(0.0, np.nan)
        result[col] = grouped[f"num_{col}"] / denom
    return result


def fetch_demographics_per_hex(
    boundary: CityBoundary,
    hex_gdf: gpd.GeoDataFrame,
    cfg: PipelineConfig,
    *,
    fetch_json: FetchJson = _fetch_json,
) -> pd.DataFrame:
    """Per-hex US demographics: h3, median_income, pct_over_65, pct_under_5.

    Non-US boundary -> empty frame instantly, no network call. Otherwise: resolve the
    TIGERweb "Census Tracts" layer, fetch tracts intersecting the hex grid's bounding box,
    pull ACS 5-year variables per (state, county), and areally interpolate onto the grid.
    Only hexes with at least one non-NaN metric are returned (features.py left-joins the
    rest). Any exception anywhere in this path is caught here — this function never raises.
    """
    if not _is_us(boundary):
        log.info(
            "%s: country=%r is not the United States — skipping demographics", boundary.name, boundary.country
        )
        return _empty()

    try:
        utm_crs = utm_crs_for(hex_gdf)
        layer_id = _find_tract_layer_id(fetch_json)
        bounds = tuple(hex_gdf.total_bounds)
        tracts = _fetch_tracts(fetch_json, layer_id, bounds)
        log.info("%s: %d TIGERweb census tracts intersect the hex grid", boundary.name, len(tracts))
        if len(tracts) == 0:
            log.warning("%s: no TIGERweb census tracts found — demographics unavailable", boundary.name)
            return _empty()

        metrics = _tract_metrics(fetch_json, tracts)
        if len(metrics) == 0:
            log.warning("%s: no ACS data for any intersecting tract — demographics unavailable", boundary.name)
            return _empty()

        per_hex = _areal_interpolate(hex_gdf, tracts, metrics, utm_crs)
        if len(per_hex) == 0:
            return _empty()

        mask = per_hex[DEMOGRAPHIC_COLS].notna().any(axis=1)
        result = per_hex.loc[mask, ["h3", *DEMOGRAPHIC_COLS]].reset_index(drop=True)
        log.info("%s: demographics attached to %d/%d hexes", boundary.name, len(result), len(hex_gdf))
        return result if len(result) else _empty()
    except Exception as exc:  # demographics are a bonus feature — never crash the pipeline
        log.warning("%s: demographics fetch failed (%s) — returning empty frame", boundary.name, exc)
        return _empty()
