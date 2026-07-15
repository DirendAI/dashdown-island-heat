# Urban Heat Islands — Dashdown dashboard

An interactive [Dashdown](https://github.com/) dashboard over the `heat-island`
pipeline: where a city is hottest, where tree canopy would cool it most, and how
well the model behind those estimates holds up.

## Run it

```bash
cd dashboard
dashdown serve .        # dev server at http://127.0.0.1:8000 (live reload)
```

## Pages

- **Overview** (`/`) — headline KPIs (hottest hex, mean LST, mean NDVI,
  high-priority hex count, model R²) and how the method works.
- **Heat map** (`/heat-map`) — hex-centroid scatter "maps" of LST and NDVI
  quintiles, plus the heat-vs-greenness relationship.
- **ML insights** (`/ml-insights`) — model quality, SHAP feature importance,
  predicted-vs-observed LST, and what spatial cross-validation buys you.
- **Planting priorities** (`/priorities`) — priority-quintile map, the top 25
  hexes to plant, and the biggest single-hex cooling win.

Every page carries a **City** selector in the filter bar and is city-agnostic —
new cities appear automatically.

## Data

All pages read `../data/heat.duckdb` (repo root `data/`, one level above this
project) via the single DuckDB source in [`sources.yaml`](sources.yaml). The path
is project-relative, so the dashboard works wherever the repo is cloned.

The database is produced by the pipeline, one city at a time, from the repo root:

```bash
uv run heat-island add-city "<city>"
```

Each run appends `cities`, `hexes`, `model_metrics`, and `feature_importance`
rows for that city; the dashboard needs no edits to show it.
