---
title: City dashboard
static_paths:
  query: SELECT city_id FROM cities ORDER BY city_id
---

```sql meta
SELECT c.name, c.country, c.n_hexes, m.r2, m.mae, m.n_train
FROM cities c
LEFT JOIN model_metrics m USING (city_id)
WHERE c.city_id = '${city_id}'
```

```sql kpis
SELECT
  MAX(mean_lst_c)                               AS hottest_hex,
  AVG(mean_lst_c)                               AS mean_lst,
  AVG(ndvi)                                     AS mean_ndvi,
  AVG(plantable_fraction)                       AS mean_plantable,
  COUNT(*) FILTER (WHERE priority_score >= 0.8) AS high_priority,
  COUNT(*)                                      AS n_hexes
FROM hexes WHERE city_id = '${city_id}'
```

```sql lst_map
SELECT h3, geometry_wkt, mean_lst_c, ndvi, priority_score
FROM hexes WHERE city_id = '${city_id}'
```

```sql ndvi_map
SELECT h3, geometry_wkt, ndvi, mean_lst_c
FROM hexes WHERE city_id = '${city_id}'
```

```sql lst_vs_ndvi
SELECT ndvi, mean_lst_c
FROM hexes WHERE city_id = '${city_id}'
```

```sql feat_imp
SELECT feature, mean_abs_shap
FROM feature_importance
WHERE city_id = '${city_id}'
ORDER BY mean_abs_shap DESC
```

```sql pred_vs_actual
SELECT mean_lst_c, predicted_lst_c
FROM hexes WHERE city_id = '${city_id}'
```

```sql prio_map
SELECT h3, geometry_wkt, priority_score, predicted_cooling_c, plantable_fraction
FROM hexes WHERE city_id = '${city_id}'
```

```sql plantable_map
SELECT h3, geometry_wkt, plantable_fraction, tree_fraction
FROM hexes WHERE city_id = '${city_id}'
```

```sql max_cooling
SELECT predicted_lst_c, predicted_cooling_c AS max_cooling, cooling_uncertainty_c
FROM hexes WHERE city_id = '${city_id}'
ORDER BY predicted_cooling_c DESC
LIMIT 1
```

```sql top25
-- Round in SQL (<Table> has no decimals attr); cooling folded with its ± spread.
-- ORDER BY the raw column so the ranking is exact despite the rounding.
SELECT
  h3,
  ROUND(mean_lst_c, 2)         AS mean_lst_c,
  ROUND(ndvi, 2)               AS ndvi,
  ROUND(tree_fraction, 2)      AS tree_fraction,
  ROUND(plantable_fraction, 2) AS plantable_fraction,
  -- COALESCE keeps the cooling value visible when uncertainty is NULL (a city
  -- migrated from v0.1 but not yet re-run) — printf would blank the whole string.
  printf('%.2f', predicted_cooling_c)
    || COALESCE(' ± ' || printf('%.2f', cooling_uncertainty_c), '') AS predicted_cooling_c,
  ROUND(priority_score, 2)     AS priority_score
FROM hexes h
WHERE h.city_id = '${city_id}'
ORDER BY h.priority_score DESC
LIMIT 25
```

# <Value data={meta} column="name" /> — heat island & planting priorities

The complete analysis for **<Value data={meta} column="name" />,
<Value data={meta} column="country" />**, pre-rendered from `data/heat.duckdb`.
This page is a single template (`pages/cities/[city_id].md`) that the static
export emits once per processed city — on the live server you also get the
interactive pages with a city selector in the filter bar.

<Ask data={kpis,meta} inline refresh=false cache_ttl=86400
  ask="Narrate this city's urban-heat story in two short paragraphs: how severe its heat island is (the hottest hex versus the city mean), how green and plantable it is, and how much confidence the model's R² warrants. Plain language, no bullet lists." />

<Grid cols=3>
<Counter data={kpis} column="hottest_hex" format="number" decimals=1 suffix="°C" label="Hottest hex" />
<Counter data={kpis} column="mean_lst" format="number" decimals=1 suffix="°C" label="City mean LST" />
<Counter data={kpis} column="mean_ndvi" format="number" decimals=2 label="Mean NDVI (greenness)" />
<Counter data={kpis} column="mean_plantable" format="number" decimals=2 label="Plantable share" />
<Counter data={kpis} column="high_priority" format="number" decimals=0 label="High-priority hexes (≥ 0.8)" />
<Counter data={kpis} column="n_hexes" format="number" decimals=0 label="Hexes analysed" />
</Grid>

## Where it's hot

Each hex is drawn at its true footprint and filled by land-surface temperature
on a **continuous** cool **blue** → amber → hot **red** ramp (Landsat summer
composite) — not quintile bands. Scroll to zoom, drag to pan; hover for the
hex's greenness and priority score.

<HexMap data={lst_map} value="mean_lst_c" unit="°C" scheme="heat"
  title="Land-surface temperature by hex" height=520 tooltip="ndvi,priority_score" />

## Where it's green

The same hexes filled by **NDVI** — tan/brown (barest ground) → **dark green**
(densest vegetation). Hover for each hex's land-surface temperature.

<HexMap data={ndvi_map} value="ndvi" scheme="greens"
  title="Vegetation (NDVI) by hex" height=520 tooltip="mean_lst_c" />

Greener hexes run cooler — the lever the planting model pulls on:

<ScatterChart data={lst_vs_ndvi} x="ndvi" y="mean_lst_c"
  height=380 title="LST vs NDVI (one point per hex)"
  explain="Describe the relationship between vegetation and surface temperature, and what the vertical spread at low NDVI means." />

## Model quality

Scored under **spatial cross-validation** (held-out spatial blocks), so these
are honest out-of-area numbers.

<Grid cols=3>
<Counter data={meta} column="r2" format="number" decimals=3 label="R² (spatial CV)" />
<Counter data={meta} column="mae" format="number" decimals=2 suffix="°C" label="Mean abs. error" />
<Counter data={meta} column="n_train" format="number" decimals=0 label="Training hexes" />
</Grid>

<BarChart data={feat_imp} x="feature" y="mean_abs_shap" horizontal height=380
  title="Feature importance (mean |SHAP|, °C)" explain />

<ScatterChart data={pred_vs_actual} x="mean_lst_c" y="predicted_lst_c"
  height=440 title="Predicted vs observed LST (°C)" explain />

## Planting priorities

Priority = **heat × achievable cooling × vulnerability**. Achievable cooling is
capped by each hex's **plantable space** (ESA WorldCover): built-up land earns a
small street-pit credit, water and existing forest count zero. Cooling figures
are **value ± uncertainty** — the spread across the 5 spatial-CV fold models.

<HexMap data={prio_map} value="priority_score" scheme="priority"
  title="Planting priority by hex" height=520 tooltip="predicted_cooling_c,plantable_fraction" />

<HexMap data={plantable_map} value="plantable_fraction" scheme="greens"
  title="Plantable land share by hex" height=520 tooltip="tree_fraction" />

### Biggest single win

<Counter data={max_cooling} column="max_cooling" format="number" decimals=2 suffix="°C" label="Max achievable cooling from greening one hex" />

The plantability-constrained figure; ± <Value data={max_cooling} column="cooling_uncertainty_c" format="number" decimals=2 />°C
across the fold models.

### Top 25 hexes to plant

<Table data={top25} title="Highest-priority hexes" />
