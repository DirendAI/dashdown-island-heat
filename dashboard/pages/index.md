---
title: Overview
sidebar_label: Overview
sidebar_position: 1
icon: "🔥"
---

```sql cities
SELECT name FROM cities ORDER BY name
```

```sql kpis
SELECT
  MAX(mean_lst_c)                              AS hottest_hex,
  AVG(mean_lst_c)                              AS mean_lst,
  AVG(ndvi)                                    AS mean_ndvi,
  COUNT(*) FILTER (WHERE priority_score >= 0.8) AS high_priority,
  COUNT(*)                                     AS n_hexes
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

```sql model_r2
SELECT r2
FROM model_metrics
WHERE city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

<Dropdown name="city" data={cities} column="name" label="City" bar />

Where a city bakes and where new tree canopy would cool it the most — mapped hex
by hex from satellite imagery and a machine-learning model. Pick a city in the
filter bar; every page re-queries for it.

## Key metrics

<Grid cols=3>
<Counter data={kpis} column="hottest_hex" format="number" decimals=1 suffix="°C" label="Hottest hex" />
<Counter data={kpis} column="mean_lst" format="number" decimals=1 suffix="°C" label="City mean LST" />
<Counter data={kpis} column="mean_ndvi" format="number" decimals=2 label="Mean NDVI (greenness)" />
<Counter data={kpis} column="high_priority" format="number" decimals=0 label="High-priority hexes (≥ 0.8)" />
<Counter data={kpis} column="n_hexes" format="number" decimals=0 label="Hexes analysed" />
<Counter data={model_r2} column="r2" format="number" decimals=3 label="Model R² (spatial CV)" />
</Grid>

## How it works

**Land-surface temperature (LST)** is a Landsat thermal composite averaged over
three summers. Vegetation, built-up and water indices (**NDVI**, **NDBI**,
**NDWI**) and **albedo** come from Sentinel-2. Every measure is aggregated onto
**H3 resolution-9 hexagons** (~0.1 km² each), so a city becomes a mosaic of
comparable cells.

A **LightGBM** regressor learns hex LST from those features under **spatial
cross-validation** — the test hexes sit in spatial blocks held out from training,
so the model can't peek at a neighbour's temperature (see
[ML insights](/ml-insights)). A **greening counterfactual** then lifts each hex's
NDVI to local park level and re-predicts LST; the drop is its modelled
`predicted_cooling_c`. Finally the planting **priority score** combines
**heat × cooling potential × vulnerability** into a single 0–1 rank
(see [Planting priorities](/priorities)).

:::note
Demographic vulnerability inputs (income, share over 65 / under 5) come from the
US Census and are absent for cities outside it, so a non-US city's priority leans
on the heat and cooling terms. The pipeline handles that automatically.
:::

## Add a city

Run this from the repository root — the pipeline downloads the imagery, builds
the hex grid, trains the model, and appends the city to `data/heat.duckdb`:

```bash
uv run heat-island add-city "<city>"
```

The dashboard is city-agnostic: the new city appears in the **City** selector on
every page on the next reload, with no edits here.
