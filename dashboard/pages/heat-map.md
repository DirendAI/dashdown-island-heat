---
title: Heat map
sidebar_label: Heat map
sidebar_position: 2
icon: "🗺️"
---

```sql cities
SELECT name FROM cities ORDER BY name
```

```sql lst_map
SELECT
  lon, lat, mean_lst_c,
  CASE NTILE(5) OVER (ORDER BY mean_lst_c)
    WHEN 1 THEN 'coolest 20%'
    WHEN 2 THEN 'cool 20%'
    WHEN 3 THEN 'moderate 20%'
    WHEN 4 THEN 'warm 20%'
    WHEN 5 THEN 'hottest 20%'
  END AS band
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
-- The final ORDER BY drives the series first-seen order = legend color order
-- (coolest→hottest maps onto the blue→red ramp). Do not remove it.
ORDER BY mean_lst_c
```

```sql ndvi_map
SELECT
  lon, lat, ndvi,
  CASE NTILE(5) OVER (ORDER BY ndvi)
    WHEN 1 THEN 'sparsest 20%'
    WHEN 2 THEN 'sparse 20%'
    WHEN 3 THEN 'moderate 20%'
    WHEN 4 THEN 'leafy 20%'
    WHEN 5 THEN 'greenest 20%'
  END AS band
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
-- Final ORDER BY = legend color order (sparsest→greenest onto brown→green). Keep it.
ORDER BY ndvi
```

```sql lst_vs_ndvi
SELECT ndvi, mean_lst_c
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

<Dropdown name="city" data={cities} column="name" label="City" bar />

Dashdown's geo maps are country/region **choropleths** joined on GeoJSON ids, not
arbitrary point maps — and a component's attributes can't be driven by a filter.
So each city "map" here is a **ScatterChart of hex centroids** (x = longitude,
y = latitude), coloured by a 5-band temperature/greenness quintile. The result is
a faithful top-down picture of the city that re-queries per selection.

## Where it's hot

Each point is one H3 hex at its true position; colour is its land-surface
temperature quintile, cool **blue** → hot **red**.

<ScatterChart data={lst_map} x="lon" y="lat" series="band"
  color="#4c78a8,#6cc5b0,#f2cf5b,#f58518,#e45756"
  height=520 title="Land-surface temperature by hex (quintile bands)" />

## Where it's green

The same hexes coloured by **NDVI** quintile. The ramp is deliberately reversed
from the heat map so the semantics read naturally: **dark green = highest NDVI**
(densest vegetation), tan/brown = barest ground.

<ScatterChart data={ndvi_map} x="lon" y="lat" series="band"
  color="#a6611a,#d8b365,#c2e699,#78c679,#238443"
  height=520 title="Vegetation (NDVI) by hex (quintile bands)" />

## Heat vs greenness

Plotting the two against each other collapses the maps into one relationship. The
expected sign is **negative** — greener hexes (higher NDVI) tend to run cooler,
which is exactly the lever the priority model pulls on.

<ScatterChart data={lst_vs_ndvi} x="ndvi" y="mean_lst_c"
  height=380 title="LST vs NDVI (one point per hex)" />

:::note
Every hex is plotted — no sampling or `LIMIT` — so the map is the full grid
(~1–2k points per city).
:::
