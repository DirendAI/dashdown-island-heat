---
title: ML insights
sidebar_label: ML insights
sidebar_position: 3
icon: "🧠"
---

```sql cities
SELECT name FROM cities ORDER BY name
```

```sql metrics
SELECT r2, mae, n_train
FROM model_metrics
WHERE city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

```sql feat_imp
SELECT feature, mean_abs_shap
FROM feature_importance
WHERE city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
ORDER BY mean_abs_shap DESC
```

```sql pred_vs_actual
SELECT mean_lst_c, predicted_lst_c
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

<Dropdown name="city" data={cities} column="name" label="City" bar />

How well the LightGBM model explains a city's heat, what drives its predictions,
and how the greening counterfactual is grounded.

## Model quality

Scored under **spatial cross-validation**, so these are honest out-of-area
numbers, not a memorised fit.

<Grid cols=3>
<Counter data={metrics} column="r2" format="number" decimals=3 label="R² (spatial CV)" />
<Counter data={metrics} column="mae" format="number" decimals=2 suffix="°C" label="Mean abs. error" />
<Counter data={metrics} column="n_train" format="number" decimals=0 label="Training hexes" />
</Grid>

## What drives the heat

Mean absolute **SHAP** value per feature — how much each input moves the
predicted temperature, on average. Built-up density (**NDBI**) and building
density dominate; vegetation and water proximity pull the other way.

<BarChart data={feat_imp} x="feature" y="mean_abs_shap" horizontal height=380
  title="Feature importance (mean |SHAP|, °C)" />

## Predicted vs observed

Each point is one hex: observed LST on the x-axis, the model's prediction on the
y-axis. A perfect model would sit on the **1:1 line** — points hug it here, and
the vertical scatter around it is the residual error (the ~1 °C MAE above).

<ScatterChart data={pred_vs_actual} x="mean_lst_c" y="predicted_lst_c"
  height=440 title="Predicted vs observed LST (°C)" />

## Why spatial CV matters

Neighbouring hexes are strongly **autocorrelated** — two adjacent cells share
almost the same temperature. A plain random train/test split would scatter test
hexes right next to training hexes, letting the model "peek" at an answer through
its neighbours and reporting an inflated score.

**Spatial cross-validation** instead holds out whole spatial **blocks** of hexes
for testing, so a test hex has no training neighbour to leak from. The R² above is
therefore a measure of genuine generalisation to **unseen ground** — the property
you need before trusting the model's cooling predictions on new areas or a new
city.
