# heat-island — Architecture & Module Contract

This document is the **binding contract** between modules. Implementation agents: implement
exactly these signatures and dataframe contracts. If you believe a contract is wrong, note it
in your final report — do **not** silently change a signature or column name.

## What this project does

Given any city name: geocode boundary → H3 hex grid (res 9) → Landsat land-surface
temperature + Sentinel-2 indices + Copernicus DEM (Microsoft Planetary Computer, keyless) →
OSM features (buildings/roads/water/parks) → optional US demographics → LightGBM model of LST
with spatial CV + SHAP → greening counterfactual (raise NDVI to park level) → priority score →
one DuckDB file (`data/heat.duckdb`) → Dashdown dashboard (`dashboard/`) with a city selector.

## Global conventions (all modules)

- Python 3.11, `uv`-managed. Run things with `uv run ...`. Tests: `uv run pytest`.
- All geometries stored/passed in **EPSG:4326** (lon/lat). Any *metric* operation (areas,
  distances, buffers) must project to the local UTM first via
  `gdf.estimate_utm_crs()` — never compute meters in 4326.
- H3 cells are **lowercase hex strings** (h3-py v4 default string form), column name `h3`.
- Every per-hex dataframe has a plain `h3` column (not index) unless stated otherwise.
- Config: a single `PipelineConfig` dataclass (`config.py`) is threaded through everything.
- Logging: `from .util import get_logger; log = get_logger(__name__)`. Log every expensive
  step at INFO with counts/timings. No `print` in library code (CLI may use rich console).
- Caching: everything expensive is cached under `cfg.cache_dir / city_id / ...` as
  parquet/geojson. Cache key includes the H3 resolution where relevant (e.g. `lst_res9.parquet`).
  A cache hit must skip network entirely.
- Retries: wrap network calls with `util.retry_call` (tenacity, 4 attempts, exponential 2s→16s).
- Determinism: seed everything from `cfg.seed` (default 42).
- **No network access in unit tests.** Tests use synthetic dataframes / geometries only.
- Errors: raise `heat_island.util.PipelineError` (or a subclass) with an actionable message
  ("City 'Xyz' geocoded to a point, not a polygon — try 'Xyz, Country'").
- Matplotlib: force `matplotlib.use("Agg")` before pyplot import (headless).

### Library API pins (versions installed; do NOT use older API names)

- **h3-py v4** (not v3!): `h3.geo_to_h3shape(geo)`, `h3.h3shape_to_cells(shape, res)`,
  `h3.cell_to_latlng(c) -> (lat, lng)`, `h3.cell_to_boundary(c) -> ((lat, lng), ...)`,
  `h3.latlng_to_cell(lat, lng, res)`, `h3.cell_to_parent(c, res)`, `h3.get_resolution(c)`.
  There is no `polyfill` / `h3_to_geo` in v4.
- **osmnx 2.x**: `ox.geocoder.geocode_to_gdf(query)`, `ox.features.features_from_polygon(poly, tags)`,
  `ox.graph.graph_from_polygon(poly, network_type=...)`, `ox.convert.graph_to_gdfs(G)`.
  Configure `ox.settings.cache_folder`, `ox.settings.http_user_agent`, `ox.settings.requests_timeout=180`,
  `ox.settings.log_console=False`. osmnx caches HTTP responses itself — point its cache at
  `cfg.cache_dir / city_id / "osm_http"`.
- **shapely 2**: `STRtree(geoms).nearest(geom)` returns the *index* into `geoms`;
  `shapely.distance` is vectorized. No `.ops.nearest_points` loops over thousands of geoms.
- **pystac-client + planetary-computer**:
  `Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)`.
- **odc-stac**: `odc.stac.load(items, bands=[...], bbox=bbox4326, crs="EPSG:4326",
  resolution=<degrees>, chunks={"x": 2048, "y": 2048}, groupby="solar_day")` → Dataset with
  `time, latitude, longitude` dims. Compute with `.compute()` after building the reduction.
- **lightgbm**: `lgb.LGBMRegressor` sklearn API.

## Repo layout

```
pyproject.toml  ARCHITECTURE.md  README.md  LICENSE
src/heat_island/
  __init__.py  config.py  util.py          # provided by architect — do not rewrite
  cli.py                                    # typer app
  boundary.py hexgrid.py satellite.py osm_features.py
  demographics.py features.py model.py simulate.py db.py
  viz.py                                    # sanity-check plotting
tests/                                      # offline unit tests
data/                                       # gitignored: heat.duckdb + cache/<city_id>/
dashboard/                                  # dashdown project (wave 3)
```

---

## config.py (PROVIDED — read it)

`PipelineConfig` dataclass; key fields: `h3_resolution=9`, `years_back=3`, `max_cloud_pct=20`,
`landsat_max_scenes=18`, `s2_max_scenes=24`, `raster_res_deg=0.00027` (~30 m),
`s2_res_deg=0.00018` (~20 m), `buffer_m=2000`, `dist_cap_m=10000.0`,
`min_water_area_m2=10000`, `cv_parent_offset=2`, `cv_folds=5`, `min_hexes=30`, `seed=42`,
`data_dir`, `db_path`, `cache_dir`, plus `FEATURES` (the canonical model feature list).

## util.py (PROVIDED — read it)

`get_logger`, `PipelineError`, `CityNotFoundError`, `DataUnavailableError`,
`retry_call(fn, *args, what="...", **kwargs)`, `slugify(text)`, `utm_crs_for(gdf_or_geom)`,
`summer_windows(lat, today) -> list[(start_iso, end_iso)]` (3 most recent completed summers;
N hemisphere Jun 1–Aug 31; S hemisphere Dec 1–Feb 28 spanning year boundary; |lat| < 10 →
whole years, tropics have no meaningful summer).

---

## boundary.py

```python
@dataclass
class CityBoundary:
    query: str            # user's original query string
    name: str             # canonical short name from Nominatim (e.g. "Gent")
    country: str          # last component of display_name, stripped (e.g. "België / Belgique / Belgien" → keep as-is)
    city_id: str          # util.slugify(f"{name}-{country_first_token}") — stable across query spellings
    geometry: BaseGeometry  # shapely (Multi)Polygon, EPSG:4326
    centroid_lat: float
    centroid_lon: float

def get_city_boundary(query: str, cfg: PipelineConfig) -> CityBoundary
```

- Use `ox.geocoder.geocode_to_gdf(query)`. First row. Must be Polygon/MultiPolygon; a
  Point/LineString → raise `CityNotFoundError` telling the user to qualify the query
  ("Springfield" → "Springfield, Illinois, USA").
- Geocode failure (no result) → `CityNotFoundError` with the query echoed.
- `country`: from `display_name` split on "," → last element stripped. For city_id take the
  first "/"-separated token of the country, slugified: `gent-belgie`. (Stable, ascii, unique
  enough; do not worry about collisions between identically-named cities in the same country.)
- Cache the boundary as GeoJSON at `cfg.cache_dir / city_id / "boundary.geojson"` **keyed by
  slug of the query** too (`boundary_<queryslug>.geojson` symlink-style duplicate is fine) so
  re-runs skip Nominatim. Simplest: first geocode → derive city_id → write cache; on rerun we
  cannot know city_id before geocoding, so also keep a tiny JSON index
  `cfg.cache_dir / "query_index.json"` mapping query-slug → city_id. Cache hit = read GeoJSON.
- Set osmnx settings (UA "heat-island-mapper/0.1 (github.com/DirendAI/dashdown-island-heat)",
  cache folder) before any osmnx call — provide a module-level `_configure_osmnx(cfg)`.

## hexgrid.py

```python
def polygon_to_hexes(geom, res: int) -> list[str]
    # h3.geo_to_h3shape(geom.__geo_interface__) — handle MultiPolygon by iterating parts and
    # unioning cell sets. Empty result (tiny polygon) → fall back to the cell containing the
    # centroid plus grid_disk(k=1) so we never return [].

def hexes_to_gdf(cells: list[str]) -> gpd.GeoDataFrame
    # columns: h3, lat, lon, geometry (shapely Polygon in 4326, ring closed, lon/lat order!)
    # lat/lon from h3.cell_to_latlng; polygon from h3.cell_to_boundary (returns (lat, lng) —
    # you MUST swap to (lng, lat) when building shapely rings).

def points_to_hex_means(df: pd.DataFrame, res: int, value_cols: list[str],
                        lat_col="lat", lon_col="lon") -> pd.DataFrame
    # assign h3 via list comprehension over zip (NOT DataFrame.apply — 3-5x slower),
    # groupby h3, mean of value_cols; returns df with h3 + value_cols. Drops NaN rows first
    # per column (use groupby mean with skipna instead of dropping whole rows).

def hex_parents(cells: Iterable[str], offset: int) -> np.ndarray
    # h3.cell_to_parent(c, res(c) - offset); offset≥res → parent res 0

def grid_stats(cells, geom) -> dict   # n_hexes, area_km2 (UTM), mean_hex_area_km2
```

## viz.py

```python
def plot_city_grid(boundary: CityBoundary, hex_gdf, out_png: Path,
                   value_col: str | None = None, title: str | None = None) -> Path
```
Boundary outline (thick), hexes (thin edges, translucent fill; if `value_col` → viridis-ish
choropleth + colorbar), equal aspect adjusted by cos(lat), title with name + hex count, save
PNG at 150 dpi. Return the path.

## satellite.py  (all three fetchers share helpers; cache under cache/<city_id>/)

STAC: `landsat-c2-l2` (platforms landsat-8/9), `sentinel-2-l2a`, `cop-dem-glo-30` from
`https://planetarycomputer.microsoft.com/api/stac/v1` with `planetary_computer.sign_inplace`.
Search per summer window from `util.summer_windows(centroid_lat)` with
`eo:cloud_cover < cfg.max_cloud_pct` (not for DEM), over `geom.bounds` expanded by ~0.01°.
Sort combined items by cloud cover ascending, cap at `cfg.landsat_max_scenes` /
`cfg.s2_max_scenes`. If zero items at <20% cloud, relax once to <40% with a WARNING; still
zero → raise `DataUnavailableError` naming the collection and windows tried.

```python
def fetch_lst_per_hex(boundary, cfg) -> pd.DataFrame        # h3, mean_lst_c
def fetch_s2_indices_per_hex(boundary, cfg) -> pd.DataFrame # h3, ndvi, ndbi, ndwi, albedo
def fetch_elevation_per_hex(boundary, cfg) -> pd.DataFrame  # h3, elevation
def fetch_landcover_per_hex(boundary, cfg) -> pd.DataFrame  # h3, plantable_fraction, tree_fraction
```

**Landsat LST** (`fetch_lst_per_hex`):
- Bands: prefer asset key `lwir11`, fall back to `ST_B10` / `lwir` (inspect first item's
  assets); QA asset `qa_pixel` (fallback `QA_PIXEL`).
- Load with odc-stac at `cfg.raster_res_deg`, EPSG:4326, bbox = city bounds + 0.005° pad.
- Mask: drop pixels where `(qa_pixel & 0b11111) != 0` (fill, dilated cloud, cirrus, cloud,
  shadow = bits 0-4) or raw DN == 0 (nodata).
- Convert: `lst_c = dn * 0.00341802 + 149.0 - 273.15`. Drop pixels outside [-20, 70] °C.
- Composite: `median(dim="time", skipna=True)` → 2D grid → dataframe of pixel centers
  (latitude/longitude coords) → `points_to_hex_means` at `cfg.h3_resolution`.
- Log: n scenes used, per-scene dates range, and composite min/mean/max °C. **Milestone
  gate: urban summer LST ~25–50 °C** — log a loud WARNING outside that.

**Sentinel-2** (`fetch_s2_indices_per_hex`):
- Bands `B02 B03 B04 B08 B11 SCL` at `cfg.s2_res_deg` (~20 m; B11 native 20 m).
- Mask: SCL in {0,1,3,8,9,10} → NaN (nodata, saturated, shadow, cloud med/high, cirrus).
- **Processing-baseline offset**: reflectance = (DN − 1000)/10000 for acquisitions on/after
  2022-01-25 (the baseline-04.00 rollout date), else DN/10000. The rule is keyed on the
  loaded time axis (a per-time offset DataArray, dims=("time",), broadcast — no pixel loops)
  rather than per-item `s2:processing_baseline`, because `groupby="solar_day"` composites
  items so per-item properties no longer map 1:1 onto time slices. Clip reflectance to [0, 1].
- Median-composite each band over time **first**, then compute indices from the composite:
  `ndvi=(B08−B04)/(B08+B04)`, `ndbi=(B11−B08)/(B11+B08)`, `ndwi=(B03−B08)/(B03+B08)`,
  `albedo = (B02+B03+B04)/3` (documented crude visible-band proxy). Guard zero denominators → NaN.
- Aggregate all four to hexes in one `points_to_hex_means` call.

**DEM** (`fetch_elevation_per_hex`): collection `cop-dem-glo-30`, asset `data`, no cloud
filter/time filter (it's static; search without datetime). Median over items if several tiles
overlap, else squeeze. Aggregate mean per hex.

**Land cover** (`fetch_landcover_per_hex`): ESA WorldCover, collection `esa-worldcover`,
asset `map` (categorical 10 m map, class codes 10..100), static — no cloud/datetime filter,
but if the search returns multiple product versions keep only the items of the LATEST
version (property `esa_worldcover:product_version`; fall back to latest item datetime).
Load at `cfg.landcover_res_deg` in EPSG:4326; class 0 / nodata → NaN. Map classes to values
**per time/tile slice first**, then median-composite, then hex-mean (never take a median of
raw categorical codes): `plantable = config.PLANTABLE_CLASS_WEIGHTS[class]` (vectorized
lookup; unknown classes → 0.0) and `tree = (class == config.LANDCOVER_TREE_CLASS)`.
`plantable_fraction` = hex mean of plantable weights ∈ [0, 1]; `tree_fraction` = hex mean of
the tree indicator ∈ [0, 1].

All four: write parquet cache `lst_res{r}.parquet` / `s2_res{r}.parquet` /
`dem_res{r}.parquet` / `landcover_res{r}.parquet`; on hit, read and return immediately
(log "cache hit"). Wrap `search().item_collection()` and `.compute()` in `retry_call`.

## osm_features.py

```python
def fetch_osm_features_per_hex(boundary, hex_gdf, cfg) -> pd.DataFrame
    # h3, building_density, road_density, dist_water_m, dist_park_m, is_park
```
- Cache parquet `osm_res{r}.parquet`.
- Work in UTM throughout (project hex_gdf + fetched features once).
- **Buildings**: `features_from_polygon(geom, {"building": True})`; keep polygonal geoms only.
  Assign each building to the hex containing its **centroid** (`latlng_to_cell` on centroid
  lat/lon in 4326 — compute centroids in UTM, then transform centroid points back to 4326).
  `building_density` = Σ building area (UTM m²) / hex area (m²), clipped to [0, 1]. Missing → 0.
- **Roads**: `graph_from_polygon(geom, network_type="drive", retain_all=True)` →
  `ox.convert.to_undirected(G)` (collapse reciprocal two-way edges — a directed graph would
  double-count them ~1.6×) → `graph_to_gdfs(G, nodes=False)`. Edge length: use the `length`
  attribute (meters). Assign
  each edge to the hex of its geometry midpoint (`.interpolate(0.5, normalized=True)` in UTM →
  back-transform). `road_density` = Σ length_m / hex_area_km² → **km per km²** (divide m by
  1000). Missing → 0. If the graph fetch fails entirely (Overpass down), log WARNING and
  return zeros for road columns rather than crashing (retry first!).
- **Water**: fetch on boundary buffered by `cfg.buffer_m` (UTM buffer → back to 4326):
  tags `{"natural": "water", "water": True, "waterway": ["riverbank"]}` → polygonal geoms with
  UTM area ≥ `cfg.min_water_area_m2`. `dist_water_m` = distance (UTM) from hex centroid to
  nearest such polygon via `STRtree`; none found → `cfg.dist_cap_m`. Cap all at `dist_cap_m`.
- **Parks**: same buffered fetch, tags `{"leisure": ["park", "garden", "nature_reserve"],
  "landuse": ["recreation_ground", "village_green", "cemetery"]}` → polygons. `dist_park_m`
  analogous. `is_park` (bool) = hex centroid within any park polygon (STRtree `query` with
  predicate "within"; boundary-exact containment is measure-zero and irrelevant here). No parks → dist capped, `is_park` all False.
- Between the 4 Overpass-backed fetches sleep ~1 s (politeness; osmnx rate-limits too).

## demographics.py (US only, optional, must NEVER crash the pipeline)

```python
def fetch_demographics_per_hex(boundary, hex_gdf, cfg, *, fetch_json=_fetch_json) -> pd.DataFrame
    # h3, median_income, pct_over_65, pct_under_5  (may be empty df with those columns)
    # fetch_json is the injectable HTTP layer (tests pass a fake; default wraps requests+retry)
```
- Gate: `country` contains "United States" (Nominatim display_name for US cities ends with
  "United States"). Else return empty frame (columns present, zero rows).
- Tract geometries: TIGERweb REST (keyless):
  `https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query`
  with `geometry=<bbox>`, `geometryType=esriGeometryEnvelope`, `inSR=4326`, `f=geojson`,
  `outFields=GEOID,STATE,COUNTY`, `spatialRel=esriSpatialRelIntersects`, paginate via
  `resultOffset`. (Layer 0 = Census Tracts; verify `?f=json` layer list at runtime and pick the
  layer named like "Census Tracts".)
- ACS 5-year (latest available vintage, try 2023 then 2022):
  `https://api.census.gov/data/{vintage}/acs/acs5?get=B19013_001E,B01001_001E,B01001_003E,B01001_027E,{65+ vars}&for=tract:*&in=state:{ss}%20county:{ccc}`
  65+ = B01001_020E..025E (male) + B01001_044E..049E (female). Optional key from env
  `CENSUS_API_KEY` appended as `&key=`. One request per (state, county) pair present in tracts.
- Areal interpolation: intersect tracts×hexes in UTM (`gpd.overlay` on the *subset* of tracts,
  hexes as the left frame); per hex, weights = intersection area; `median_income` =
  area-weighted mean (ignore tracts with sentinel negatives like -666666666 → NaN);
  `pct_over_65 = Σ w·(over65/total) / Σ w` etc. (rates in **percent 0–100**).
- ANY exception → log WARNING with the reason, return empty frame. Unit tests mock the HTTP
  layer (inject a `fetch_json` callable parameter with default = real one).

## features.py (assembly)

```python
def build_feature_table(boundary, cfg) -> gpd.GeoDataFrame
```
- Canonical grid = `polygon_to_hexes(boundary.geometry, cfg.h3_resolution)` → `hexes_to_gdf`.
- Left-join satellite (LST, S2, DEM, land cover), OSM, demographics frames onto the grid by `h3`.
- Drop hexes with NaN `mean_lst_c` **or** NaN `ndvi` (satellite coverage gaps) — log the count.
- Fills: `elevation` → city median; `dist_water_m`/`dist_park_m` → `cfg.dist_cap_m`;
  `building_density`/`road_density` → 0; `ndbi`/`ndwi`/`albedo` → city median.
  `plantable_fraction`/`tree_fraction`: per-row NaN → city median; if the land-cover fetch
  failed entirely (columns absent) → plantable 1.0 (unconstrained legacy) / tree 0.0 + WARNING.
  Demographics stay NaN (nullable). `is_park` → False.
- Enforce `cfg.min_hexes` (else `PipelineError`: city too small / no coverage).
- Returns GeoDataFrame with: h3, lat, lon, geometry, is_park, all 9 model features
  (`config.FEATURES` order), plantable_fraction, tree_fraction, mean_lst_c, and the 3
  demographic columns. (`plantable_fraction`/`tree_fraction` are intervention-layer inputs,
  deliberately NOT in `FEATURES`: tree cover's thermal effect is already carried by NDVI, and
  keeping FEATURES stable keeps SHAP comparable across versions.)

## model.py

```python
FEATURES = config.FEATURES  # ["ndvi","ndbi","ndwi","albedo","elevation",
                            #  "building_density","road_density","dist_water_m","dist_park_m"]

@dataclass
class ModelResult:
    model: lgb.LGBMRegressor      # final model fit on ALL rows
    r2: float                     # out-of-fold spatial-CV R²
    mae: float                    # out-of-fold MAE (°C)
    n_train: int
    shap_importance: dict[str, float]   # feature -> mean |SHAP|
    fold_models: list             # the k models trained on CV folds (fold order) —
                                  # kept so simulate.py can spread the cooling estimate

def train_and_evaluate(df: pd.DataFrame, cfg: PipelineConfig) -> ModelResult
```
- X = df[FEATURES], y = df["mean_lst_c"].
- **Spatial CV**: groups = the H3 parent of each hex at `res - cfg.cv_parent_offset`
  (res-9 → res-7 blocks ≈ 5 km). model.py computes this locally (`_spatial_groups`,
  equivalent to `hexgrid.hex_parents`) so it stays import-independent of hexgrid. `GroupKFold(n_splits=min(cfg.cv_folds, n_unique_groups))`; if <2 groups, plain
  `KFold(2)` + WARNING. Collect out-of-fold predictions → single global R² and MAE.
- Params (fixed): `n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=cfg.seed, n_jobs=-1,
  verbosity=-1` (`subsample_freq=1` is required or LightGBM leaves row bagging off).
- Final model refit on all rows. SHAP: `shap.TreeExplainer(final_model).shap_values(X)` →
  `mean(|values|, axis=0)` per feature.
- Keep the module free of city-specific logic so a global multi-city model can replace it
  (train_and_evaluate takes any df with FEATURES + target).

## simulate.py

```python
def greening_target_ndvi(df, cfg) -> float
    # 75th percentile of ndvi among is_park hexes if ≥ 20 park hexes,
    # else 90th percentile of all ndvi (fallback, log it).
    # An absent is_park column counts as zero park hexes (fallback path, no crash).

def run_greening(df, model, cfg, fold_models=None) -> pd.DataFrame
    # adds predicted_lst_c, predicted_cooling_c, cooling_uncertainty_c
```
- `predicted_lst_c = model.predict(X)` (observed features).
- **Plantability-constrained counterfactual**: a hex can only close the gap to the greening
  target in proportion to its plantable share:
  `gap = clip(target - ndvi, 0, None)`; `ndvi_cf = ndvi + plantable_fraction * gap`
  (never lowers NDVI; plantable 0 — water, solid canopy — → no change → zero cooling; a
  missing `plantable_fraction` column → 1.0 = the old unconstrained behaviour, with WARNING;
  per-row NaN plantable → 1.0 for that row). `predicted_cooling_c = clip(predicted_lst_c -
  model.predict(X_cf), 0, None)`.
- **Uncertainty**: `cooling_uncertainty_c` = std across `fold_models` of
  `(model_i.predict(X) - model_i.predict(X_cf))` (unclipped deltas — the honest spread of
  the spatial-CV ensemble). `fold_models` None/empty → NaN column.

```python
def compute_priority(df) -> pd.DataFrame           # adds priority_score in [0, 1]
```
- `heat_rank` = percentile rank (0–1, `rank(pct=True)`) of `mean_lst_c`.
- `cooling_norm` = predicted_cooling / max(predicted_cooling); all-zero → zeros + WARNING.
- `vulnerability`: if ALL of median_income/pct_over_65/pct_under_5 are NaN for a row → 1.0.
  Else v_raw = mean of available of [1−rank(income), rank(pct_over_65), rank(pct_under_5)]
  (ranks pct=True within city); vulnerability = 0.5 + 0.5·v_raw ∈ [0.5, 1].
- `raw = heat_rank * cooling_norm * vulnerability`; `priority_score = raw / raw.max()`
  (max 0 → all zeros). Guaranteed within [0, 1].
- Deliberately **no** separate plantability multiplier here: feasibility already enters
  through the constrained cooling (a zero-plantable hex has zero cooling and hence zero
  priority); multiplying by `plantable_fraction` again would double-penalize dense cores.

## db.py (dashboard contract — EXACT schema)

```sql
CREATE TABLE IF NOT EXISTS cities(
  city_id TEXT PRIMARY KEY, name TEXT NOT NULL, country TEXT,
  centroid_lat DOUBLE, centroid_lon DOUBLE, processed_at TIMESTAMP, n_hexes INTEGER);
CREATE TABLE IF NOT EXISTS hexes(
  city_id TEXT, h3 TEXT, lat DOUBLE, lon DOUBLE, geometry_wkt TEXT,
  mean_lst_c DOUBLE, ndvi DOUBLE, ndbi DOUBLE, ndwi DOUBLE, albedo DOUBLE,
  elevation DOUBLE, building_density DOUBLE, road_density DOUBLE,
  dist_water_m DOUBLE, dist_park_m DOUBLE,
  plantable_fraction DOUBLE, tree_fraction DOUBLE,
  median_income DOUBLE, pct_over_65 DOUBLE, pct_under_5 DOUBLE,
  predicted_lst_c DOUBLE, predicted_cooling_c DOUBLE, cooling_uncertainty_c DOUBLE,
  priority_score DOUBLE,
  PRIMARY KEY (city_id, h3));
CREATE TABLE IF NOT EXISTS model_metrics(
  city_id TEXT PRIMARY KEY, r2 DOUBLE, mae DOUBLE, n_train INTEGER, trained_at TIMESTAMP);
CREATE TABLE IF NOT EXISTS feature_importance(
  city_id TEXT, feature TEXT, mean_abs_shap DOUBLE, PRIMARY KEY (city_id, feature));
```

```python
def connect(db_path) -> duckdb.DuckDBPyConnection   # mkdir parents, create tables
def upsert_city(db_path, boundary, hex_df, metrics) -> None
    # boundary/metrics are duck-typed (BoundaryLike/MetricsLike Protocols): any objects with
    # city_id/name/country/centroid_* and r2/mae/n_train/shap_importance attributes — db.py
    # deliberately imports neither boundary.py nor model.py.
    # single transaction: DELETE all four tables WHERE city_id = ?, then INSERT.
    # geometry_wkt from hex geometry .wkt. n_hexes = len(hex_df). processed_at/trained_at = now UTC.
def list_cities(db_path) -> pd.DataFrame            # cities joined with metrics
def remove_city(db_path, city_id) -> int            # rows removed from hexes; all tables cleaned
```
`upsert_city` accepts the enriched hex GeoDataFrame and selects exactly the schema columns
(missing demographic cols → NULL). Round floats sensibly (e.g. 4 decimals for indices, 2 for °C).

## cli.py (typer + rich)

```
heat-island add-city "Berlin, Germany" [--resolution 9] [--force]  # --force ignores caches
heat-island list-cities
heat-island remove-city <city_id>
heat-island preview "Ghent, Belgium" [--resolution 9] [--out out/ghent.png]   # milestone-1 sanity
```
- `add-city` pipeline order: boundary → grid → LST → S2 → DEM → OSM → demographics →
  build_feature_table → train_and_evaluate → run_greening → compute_priority → upsert_city.
  Rich progress/status lines per step; final rich table: n_hexes, LST min/mean/max, R², MAE,
  top-3 SHAP features, max predicted cooling, #hexes with priority ≥ 0.8.
- `preview`: boundary + grid + `viz.plot_city_grid` + print `grid_stats` (n_hexes, area).
- Friendly failure: catch `PipelineError` → red message, exit 1 (no traceback spam).

## Dashboard (wave 3 — see dashboard/AGENTS.md when scaffolded)

Reads `data/heat.duckdb` via `sources.yaml`. Every page: city dropdown fed by
`SELECT city_id, name FROM cities ORDER BY name`, all queries filtered by the selection.
Pages: Overview (counters + method), Heat map (LST map, NDVI map, LST-vs-NDVI scatter),
ML insights (importance bars, R²/MAE counters, predicted-vs-actual scatter),
Planting priorities (priority map + top-25 table). Verify with `dashdown check` +
`dashdown screenshot`. Map: check component capabilities first — polygons via geometry_wkt
if supported, else lat/lon points colored by metric.

## Milestone gates (verified by the architect before each next wave)

1. Ghent grid: ~1300–1800 res-9 hexes, plot looks like Ghent, tests pass.
2. Ghent LST composite: min/mean/max within ~15–55 °C, mean roughly 25–40 °C.
3. Features: <10% hexes dropped; NDVI in [-0.2, 0.95]; densities plausible.
4. Model: spatial-CV R² ≥ ~0.4 (typically 0.6–0.85); NDVI/NDBI among top SHAP features.
5. DuckDB: all 4 tables populated; re-running add-city doesn't duplicate rows.
6. Dashboard: `dashdown check` clean; screenshots show data for ≥ 2 cities via selector.
