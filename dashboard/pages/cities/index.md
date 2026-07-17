---
title: Cities
sidebar_label: City pages
sidebar_position: 5
icon: "🏙️"
---

```sql cities_dir
SELECT c.city_id, c.name, c.country, c.n_hexes,
       ROUND(m.r2, 3) AS model_r2
FROM cities c
LEFT JOIN model_metrics m USING (city_id)
ORDER BY c.name
```

# City snapshot pages

Every processed city gets a standalone page with the complete analysis —
heat maps, model quality, and planting priorities — pre-rendered by the static
export. Click a row:

<Table data={cities_dir} row_link="/cities/{city_id}" title="Cities in data/heat.duckdb" />

To add a city, run `uv run heat-island add-city "<city>"` from the repository
root, then rebuild the static site (`cd dashboard && dashdown build . --out ../site`).
