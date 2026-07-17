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
  plantable share, high-priority hex count, hexes analysed, model R²) and how
  the method works, including the plantability-constrained cooling step.
- **Heat map** (`/heat-map`) — zoomable hex-polygon maps of LST and NDVI
  (wheel to zoom, drag to pan), plus the heat-vs-greenness relationship.
- **ML insights** (`/ml-insights`) — model quality, SHAP feature importance,
  predicted-vs-observed LST, what spatial cross-validation buys you, and the
  fold-model ensemble spread behind each cooling estimate's ± uncertainty.
- **Planting priorities** (`/priorities`) — zoomable priority and
  plantable-space maps ("where trees can go"), the top 25 hexes to
  plant (cooling shown as value ± uncertainty, plus existing-canopy and
  plantable-space columns), and the biggest achievable single-hex cooling win.

Every page carries a **City** selector in the filter bar and is city-agnostic —
new cities appear automatically.

## Custom components & AI

- `components/HexMap/` — the zoomable hex map (ECharts custom series over the
  real H3 polygon geometries, inside-type dataZoom for wheel-zoom + drag-pan).
- AI commentary uses Dashdown's `<Ask>` component and chart `explain` attribute
  with the Mistral provider configured in `dashdown.yaml`. Set
  `MISTRAL_API_KEY` in the environment to enable it (locally: `export
  MISTRAL_API_KEY=...`; in CI: the repo secret). Without the key everything
  else works and AI cards show a muted note. Static builds bake the answers at
  build time.

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
