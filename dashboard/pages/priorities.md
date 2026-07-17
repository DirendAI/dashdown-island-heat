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
SELECT h3, geometry_wkt, priority_score, predicted_cooling_c, plantable_fraction
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

```sql plantable_map
SELECT h3, geometry_wkt, plantable_fraction, tree_fraction
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
```

```sql top25
-- Round to 2 dp in SQL: <Table> has no `decimals` attr (only per-column
-- `format=` type + currency/locale/date_format), so precision is set here.
-- predicted_cooling_c is folded together with its ± spread (cooling_uncertainty_c)
-- into one printf'd "value ± uncertainty" string, matching how the figure reads
-- in prose everywhere else on this page.
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
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
ORDER BY h.priority_score DESC
LIMIT 25
```

```sql max_cooling
-- The row for the single hottest-cooling hex, paired with that same hex's own
-- uncertainty (not some independent city-wide average).
SELECT
  predicted_cooling_c AS max_cooling,
  cooling_uncertainty_c
FROM hexes h
WHERE h.city_id = COALESCE(
  (SELECT MIN(city_id) FROM cities WHERE name = '${city}'),
  (SELECT MIN(city_id) FROM cities))
ORDER BY predicted_cooling_c DESC
LIMIT 1
```

<Dropdown name="city" data={cities} column="name" label="City" bar />

**Plant trees where it's hot, achievably coolable, and vulnerable.** The
priority score multiplies a hex's observed heat by its **achievable** cooling
potential and its neighbourhood vulnerability, so the highest-ranked cells are
where new canopy removes the most degrees from the places least able to cope.
Achievable cooling is constrained by **plantable space**: the share of each
hex's land cover (ESA WorldCover) that could physically take new canopy, with
built-up land earning a small street-pit/depaving credit and water or existing
forest counting zero. Cooling figures throughout this page are shown as
**value ± uncertainty**, the spread of the estimate across the 5 spatial-CV
fold models.

## Priority map

Hexes drawn at their true footprint and filled by a **continuous** priority
gradient, low **blue** → orange → highest **red**. Red clusters are the first
places to send crews. **Scroll to zoom, drag to pan**; hover a hex for its
predicted cooling and plantable share.

<HexMap data={prio_map} value="priority_score" scheme="priority"
  title="Planting priority by hex" height=520 tooltip="predicted_cooling_c,plantable_fraction" />

## Where trees can go

The same hexes filled by **plantable-space share** — how much of each hex's land
cover (ESA WorldCover, 10 m) could physically take new canopy. Grass, shrub,
cropland and bare ground count fully; built-up land earns a small **street-pit /
depaving credit** (0.15); water and existing forest count zero — there's nowhere
left to plant. Brown hexes have little room left; **dark green** hexes are the
most open canvas. Hover for each hex's existing canopy (`tree_fraction`).

<HexMap data={plantable_map} value="plantable_fraction" scheme="greens"
  title="Plantable land share by hex" height=520 tooltip="tree_fraction" />

## Biggest single win

<Counter data={max_cooling} column="max_cooling" format="number" decimals=2 suffix="°C" label="Max achievable cooling from greening one hex" />

This is the **plantability-constrained** figure — a hex sitting on solid
existing forest or water always scores zero here, however hot it runs. The
spread — ± <Value data={max_cooling} column="cooling_uncertainty_c" format="number" decimals=2 />°C
for this hex — is how much the estimate varies across the 5 spatial-CV fold
models, not a formal confidence interval.

## Top 25 hexes to plant

Ranked by priority score. `predicted_cooling_c` is shown as **value ±
uncertainty**: the °C drop the model expects once the hex's plantable share is
greened to local park level, followed by the spread of that estimate across
the 5 spatial-CV fold models. `tree_fraction` is existing canopy share;
`plantable_fraction` is how much of the hex could still take new trees.

<Table data={top25} title="Highest-priority hexes" />

<Ask data={top25,max_cooling} label="Where to plant first" refresh=false cache_ttl=86400
  ask="Summarize where the highest-priority hexes cluster and what they have in common (LST, NDVI, plantable share). End with the single biggest achievable cooling win. Three sentences." />
