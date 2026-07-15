---
title: Planting priorities
sidebar_label: Planting priorities
sidebar_position: 4
icon: "🌳"
---

```sql cities
SELECT name FROM cities ORDER BY name
```

```sql prio_map
SELECT
  lon, lat, priority_score,
  CASE NTILE(5) OVER (ORDER BY priority_score)
    WHEN 1 THEN 'lowest 20%'
    WHEN 2 THEN 'low 20%'
    WHEN 3 THEN 'moderate 20%'
    WHEN 4 THEN 'high 20%'
    WHEN 5 THEN 'highest 20%'
  END AS band
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
-- Final ORDER BY = legend color order (lowest→highest onto blue→red). Keep it.
ORDER BY priority_score
```

```sql top25
-- Round to 2 dp in SQL: <Table> has no `decimals` attr (only per-column
-- `format=` type + currency/locale/date_format), so precision is set here.
-- ORDER BY the raw column so the ranking is exact despite the rounding.
SELECT
  h3,
  ROUND(mean_lst_c, 2)          AS mean_lst_c,
  ROUND(ndvi, 2)                AS ndvi,
  ROUND(predicted_cooling_c, 2) AS predicted_cooling_c,
  ROUND(priority_score, 2)      AS priority_score
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
ORDER BY h.priority_score DESC
LIMIT 25
```

```sql max_cooling
SELECT MAX(predicted_cooling_c) AS max_cooling
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

<Dropdown name="city" data={cities} column="name" label="City" bar />

**Plant trees where it's hot, coolable, and vulnerable.** The priority score
multiplies a hex's observed heat by its modelled cooling potential and its
neighbourhood vulnerability, so the highest-ranked cells are where new canopy
removes the most degrees from the places least able to cope.

## Priority map

Hex centroids coloured by **priority quintile**, low **blue** → highest **red**.
Red clusters are the first places to send crews.

<ScatterChart data={prio_map} x="lon" y="lat" series="band"
  color="#4c78a8,#6cc5b0,#f2cf5b,#f58518,#e45756"
  height=520 title="Planting priority by hex (quintile bands)" />

## Biggest single win

<Counter data={max_cooling} column="max_cooling" format="number" decimals=2 suffix="°C" label="Max modelled cooling from greening one hex" />

## Top 25 hexes to plant

Ranked by priority score. `predicted_cooling_c` is the °C drop the model expects
if that hex's greenery were raised to local park level.

<Table data={top25} title="Highest-priority hexes" sort="priority_score desc" />
