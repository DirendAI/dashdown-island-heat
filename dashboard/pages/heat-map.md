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
SELECT h3, geometry_wkt, mean_lst_c, ndvi, priority_score
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

```sql ndvi_map
SELECT h3, geometry_wkt, ndvi, mean_lst_c
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

```sql lst_vs_ndvi
SELECT ndvi, mean_lst_c
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

<Dropdown name="city" data={cities} column="name" label="City" bar />

Each city map is a true hex-polygon mosaic drawn from every hex's own footprint
(`geometry_wkt`, EPSG:4326), coloured by a **continuous** gradient over the
metric rather than 5 quintile bands — so equal colours mean equal values across
cities, not just equal rank within one. **Scroll to zoom, drag to pan** (pinch
on touch); hover a hex for its exact numbers. The map re-queries whenever you
change the City filter.

## Where it's hot

Every hex sits at its true position and shape, filled by land-surface
temperature on a cool **blue** → amber → hot **red** ramp. Hover for the hex's
greenness (NDVI) and planting-priority score.

<HexMap data={lst_map} value="mean_lst_c" unit="°C" scheme="heat"
  title="Land-surface temperature by hex" height=520 tooltip="ndvi,priority_score" />

## Where it's green

The same hexes filled by **NDVI**. The ramp runs tan/brown (barest ground) →
**dark green** (densest vegetation), so the semantics read naturally and invert
the heat map above. Hover for each hex's land-surface temperature.

<HexMap data={ndvi_map} value="ndvi" scheme="greens"
  title="Vegetation (NDVI) by hex" height=520 tooltip="mean_lst_c" />

:::note
NDVI blends all vegetation together. The v0.2 planting model splits it
further — existing canopy (`tree_fraction`) and open, plantable ground
(`plantable_fraction`) are broken out on [Planting priorities](/priorities).
:::

## Heat vs greenness

Plotting the two against each other collapses the maps into one relationship. The
expected sign is **negative** — greener hexes (higher NDVI) tend to run cooler,
which is exactly the lever the priority model pulls on.

<ScatterChart data={lst_vs_ndvi} x="ndvi" y="mean_lst_c"
  height=380 title="LST vs NDVI (one point per hex)"
  explain="Describe the relationship between vegetation and surface temperature, and what the vertical spread at low NDVI means." />

:::note
Every hex is plotted — no sampling or `LIMIT` — so the map is the full grid
(~1.7–2.6k hexes per city).
:::
