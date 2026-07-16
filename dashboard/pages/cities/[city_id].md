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
SELECT
  lon, lat,
  CASE NTILE(5) OVER (ORDER BY mean_lst_c)
    WHEN 1 THEN 'coolest 20%'
    WHEN 2 THEN 'cool 20%'
    WHEN 3 THEN 'moderate 20%'
    WHEN 4 THEN 'warm 20%'
    WHEN 5 THEN 'hottest 20%'
  END AS band
FROM hexes WHERE city_id = '${city_id}'
-- Final ORDER BY = legend color order (coolest→hottest onto blue→red). Keep it.
ORDER BY mean_lst_c
```

```sql ndvi_map
SELECT
  lon, lat,
  CASE NTILE(5) OVER (ORDER BY ndvi)
    WHEN 1 THEN 'sparsest 20%'
    WHEN 2 THEN 'sparse 20%'
    WHEN 3 THEN 'moderate 20%'
    WHEN 4 THEN 'leafy 20%'
    WHEN 5 THEN 'greenest 20%'
  END AS band
FROM hexes WHERE city_id = '${city_id}'
-- Final ORDER BY = legend color order (sparsest→greenest onto brown→green). Keep it.
ORDER BY ndvi
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
SELECT
  lon, lat,
  CASE NTILE(5) OVER (ORDER BY priority_score)
    WHEN 1 THEN 'lowest 20%'
    WHEN 2 THEN 'low 20%'
    WHEN 3 THEN 'moderate 20%'
    WHEN 4 THEN 'high 20%'
    WHEN 5 THEN 'highest 20%'
  END AS band
FROM hexes WHERE city_id = '${city_id}'
-- Final ORDER BY = legend color order (lowest→highest onto blue→red). Keep it.
ORDER BY priority_score
```

```sql plantable_map
SELECT
  lon, lat,
  CASE NTILE(5) OVER (ORDER BY plantable_fraction)
    WHEN 1 THEN 'least plantable 20%'
    WHEN 2 THEN 'low 20%'
    WHEN 3 THEN 'moderate 20%'
    WHEN 4 THEN 'high 20%'
    WHEN 5 THEN 'most plantable 20%'
  END AS band
FROM hexes WHERE city_id = '${city_id}'
-- Final ORDER BY = legend color order (least→most plantable onto grey→green). Keep it.
ORDER BY plantable_fraction
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

<Grid cols=3>
<Counter data={kpis} column="hottest_hex" format="number" decimals=1 suffix="°C" label="Hottest hex" />
<Counter data={kpis} column="mean_lst" format="number" decimals=1 suffix="°C" label="City mean LST" />
<Counter data={kpis} column="mean_ndvi" format="number" decimals=2 label="Mean NDVI (greenness)" />
<Counter data={kpis} column="mean_plantable" format="number" decimals=2 label="Plantable share" />
<Counter data={kpis} column="high_priority" format="number" decimals=0 label="High-priority hexes (≥ 0.8)" />
<Counter data={kpis} column="n_hexes" format="number" decimals=0 label="Hexes analysed" />
</Grid>

## Where it's hot

Each point is one H3 hex at its true position; colour is its land-surface
temperature quintile, cool **blue** → hot **red** (Landsat summer composite).

<ScatterChart data={lst_map} x="lon" y="lat" series="band"
  color="#4c78a8,#6cc5b0,#f2cf5b,#f58518,#e45756"
  height=520 title="Land-surface temperature by hex (quintile bands)" />

## Where it's green

The same hexes coloured by **NDVI** quintile — dark green = densest vegetation,
tan/brown = barest ground.

<ScatterChart data={ndvi_map} x="lon" y="lat" series="band"
  color="#a6611a,#d8b365,#c2e699,#78c679,#238443"
  height=520 title="Vegetation (NDVI) by hex (quintile bands)" />

Greener hexes run cooler — the lever the planting model pulls on:

<ScatterChart data={lst_vs_ndvi} x="ndvi" y="mean_lst_c"
  height=380 title="LST vs NDVI (one point per hex)" />

## Model quality

Scored under **spatial cross-validation** (held-out spatial blocks), so these
are honest out-of-area numbers.

<Grid cols=3>
<Counter data={meta} column="r2" format="number" decimals=3 label="R² (spatial CV)" />
<Counter data={meta} column="mae" format="number" decimals=2 suffix="°C" label="Mean abs. error" />
<Counter data={meta} column="n_train" format="number" decimals=0 label="Training hexes" />
</Grid>

<BarChart data={feat_imp} x="feature" y="mean_abs_shap" horizontal height=380
  title="Feature importance (mean |SHAP|, °C)" />

<ScatterChart data={pred_vs_actual} x="mean_lst_c" y="predicted_lst_c"
  height=440 title="Predicted vs observed LST (°C)" />

## Planting priorities

Priority = **heat × achievable cooling × vulnerability**. Achievable cooling is
capped by each hex's **plantable space** (ESA WorldCover): built-up land earns a
small street-pit credit, water and existing forest count zero. Cooling figures
are **value ± uncertainty** — the spread across the 5 spatial-CV fold models.

<ScatterChart data={prio_map} x="lon" y="lat" series="band"
  color="#4c78a8,#6cc5b0,#f2cf5b,#f58518,#e45756"
  height=520 title="Planting priority by hex (quintile bands)" />

<ScatterChart data={plantable_map} x="lon" y="lat" series="band"
  color="#bdbdbd,#c7e9c0,#a1d99b,#41ab5d,#006d2c"
  height=520 title="Plantable land share by hex (quintile bands)" />

### Biggest single win

<Counter data={max_cooling} column="max_cooling" format="number" decimals=2 suffix="°C" label="Max achievable cooling from greening one hex" />

The plantability-constrained figure; ± <Value data={max_cooling} column="cooling_uncertainty_c" format="number" decimals=2 />°C
across the fold models.

### Top 25 hexes to plant

<Table data={top25} title="Highest-priority hexes" />
