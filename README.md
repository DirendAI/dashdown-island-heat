# heat-island 🌡️🌳

Map urban heat islands for **any city in the world**, use machine learning to predict where
tree planting would cool the city most, and explore the results in a
[Dashdown](https://pypi.org/project/dashdown-md/) dashboard.

Everything runs on **free, keyless data**: Landsat land-surface temperature and Sentinel-2
vegetation indices from the Microsoft Planetary Computer, Copernicus DEM elevation, and
OpenStreetMap buildings/roads/water/parks. The only optional key is a US Census API key for
demographic vulnerability weighting of US cities.

> Docs in progress — full setup, method notes, and screenshots land at the end of the build.
> See `ARCHITECTURE.md` for the module contract.

## Quick start

```bash
uv sync
uv run heat-island preview "Ghent, Belgium"       # sanity-check boundary + hex grid
uv run heat-island add-city "Ghent, Belgium"      # full pipeline → data/heat.duckdb
uv tool install dashdown-md
cd dashboard && dashdown serve                    # dashboard on http://localhost:8501
```

## License

MIT
