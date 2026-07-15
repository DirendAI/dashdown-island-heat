# heat-island 🌡️🌳

Map urban heat islands for **any city in the world**, use machine learning to predict where
tree planting would cool the city most, and explore the results in a
[Dashdown](https://pypi.org/project/dashdown-md/) dashboard.

Everything runs on **free, keyless data**:

| Layer | Source | Access |
| ----- | ------ | ------ |
| Land surface temperature | Landsat 8/9 Collection 2 Level-2 (`ST_B10`/`lwir11`) | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC, no key |
| Vegetation / built-up / water / albedo | Sentinel-2 L2A (NDVI, NDBI, NDWI, visible-band albedo proxy) | Planetary Computer STAC, no key |
| Elevation | Copernicus DEM GLO-30 | Planetary Computer STAC, no key |
| Buildings, roads, water bodies, parks | OpenStreetMap | Nominatim + Overpass, no key |
| Demographics (US cities only, optional) | Census ACS 5-year + TIGERweb | keyless (optional `CENSUS_API_KEY` env var raises rate limits) |

## How it works

1. **Boundary** — the city name is geocoded with Nominatim (via OSMnx) and its administrative
   polygon fetched. Works for any city worldwide; no US-specific geometries anywhere in the
   core pipeline.
2. **Hex grid** — the polygon is tiled with [H3](https://h3geo.org/) hexagons (resolution 9,
   ~0.1 km², configurable), so every city becomes comparable.
3. **Heat** — Landsat surface temperature scenes from the last 3 summers (hemisphere-aware,
   < 20 % cloud) are cloud-masked with `QA_PIXEL`, converted to °C
   (`DN × 0.00341802 + 149.0 − 273.15`), median-composited, and averaged per hexagon.
4. **Surface features** — a cloud-masked (SCL) Sentinel-2 median composite yields NDVI, NDBI,
   NDWI and a visible-band albedo proxy per hexagon (with the processing-baseline ≥ 04.00
   reflectance offset handled); Copernicus DEM adds elevation; OSM adds building footprint
   density, road density, distance to large water, and distance to parks.
5. **Model** — a LightGBM regression predicts hexagon LST from the 9 surface/urban-form
   features. Validation uses **spatial block cross-validation** (GroupKFold over coarser H3
   parent cells) so spatial autocorrelation can't leak between folds. SHAP values explain
   what drives the heat.
6. **Greening counterfactual** — every hexagon's NDVI is raised to the 75th percentile of the
   city's park hexagons (never lowered) and the model re-predicts LST. The drop, floored at
   zero, is `predicted_cooling_c`: *the modeled payoff of greening that block*.
7. **Priority score** — `heat percentile × normalized cooling × vulnerability` (vulnerability
   uses income/age demographics for US cities, defaults to 1 elsewhere), normalized to [0, 1].
8. Everything lands in one DuckDB file (`data/heat.duckdb`) that the Dashdown dashboard reads.

## Quick start

```bash
# 1. pipeline (uv-managed)
uv sync
uv run heat-island preview "Ghent, Belgium"     # sanity check: boundary + hex grid plot
uv run heat-island add-city "Ghent, Belgium"    # full pipeline (~10-20 min on first run)
uv run heat-island add-city "Ljubljana"         # any city — one command each
uv run heat-island list-cities

# 2. dashboard
uv tool install dashdown-md
cd dashboard && dashdown serve                  # → http://127.0.0.1:8000
```

Re-running `add-city` is fast: satellite composites and OSM responses are cached under
`data/cache/<city_id>/`. Use `--force` to refetch, `--resolution 8` for a coarser/faster grid.

### CLI

| Command | What it does |
| ------- | ------------ |
| `heat-island add-city "<city>"` | full pipeline → upsert into `data/heat.duckdb` |
| `heat-island list-cities` | processed cities with model metrics |
| `heat-island remove-city <city_id>` | delete a city's rows (cache kept) |
| `heat-island preview "<city>"` | boundary + hex grid sanity plot, no downloads |

## Dashboard

The Dashdown project in `dashboard/` serves four pages, each with a **city selector** — every
processed city is instantly explorable, and adding a brand-new one is a single
`heat-island add-city` away:

- **Overview** — headline counters (hottest hex, city mean LST, mean NDVI, high-priority hex
  count) and a method explainer.
- **Heat map** — hexagons colored by surface temperature, a vegetation view, and the
  LST-vs-NDVI relationship.
- **ML insights** — SHAP feature importance, spatial-CV R²/MAE, predicted-vs-actual scatter.
- **Planting priorities** — the greening priority map and a top-25 table of where trees pay
  off most.

## Screenshots

_Generated at build time — see `docs/screenshots/`._

## DuckDB schema (the pipeline ↔ dashboard contract)

```
cities(city_id, name, country, centroid_lat, centroid_lon, processed_at, n_hexes)
hexes(city_id, h3, lat, lon, geometry_wkt, mean_lst_c, ndvi, ndbi, ndwi, albedo,
      elevation, building_density, road_density, dist_water_m, dist_park_m,
      median_income, pct_over_65, pct_under_5,
      predicted_lst_c, predicted_cooling_c, priority_score)
model_metrics(city_id, r2, mae, n_train, trained_at)
feature_importance(city_id, feature, mean_abs_shap)
```

## Project layout

```
src/heat_island/      # pipeline: boundary, hexgrid, satellite, osm_features,
                      #           demographics, features, model, simulate, db, cli
tests/                # offline unit tests (indices, scoring, schema contract)
data/                 # heat.duckdb + per-city caches (gitignored)
dashboard/            # dashdown project (pages/, sources.yaml)
ARCHITECTURE.md       # binding module contract
```

Run the tests with `uv run pytest`.

## Caveats & honest limitations

- **LST ≠ air temperature.** Landsat measures skin/surface temperature on clear summer
  mid-mornings; it exaggerates rooftop/asphalt extremes relative to what a pedestrian feels,
  but ranks hot spots well.
- **The greening counterfactual is model-based.** Raising NDVI in feature space approximates
  "plant trees here"; it inherits the model's blind spots (e.g. irrigation, building shade).
- **Albedo is a crude proxy** (mean visible reflectance), not a BRDF-corrected product.
- Cities whose OSM coverage is thin will have noisier building/road features.
- Tropical cities (|lat| < 10°) composite over whole years instead of "summer".

## License

MIT — data sources retain their own licenses (Landsat/Sentinel: public domain / open;
OSM: ODbL; Census: public domain).
