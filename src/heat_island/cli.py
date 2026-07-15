"""heat-island command line interface (typer + rich).

Entry point: `heat-island = "heat_island.cli:app"` (see pyproject.toml).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .boundary import get_city_boundary
from .config import PipelineConfig
from .hexgrid import grid_stats, hexes_to_gdf, polygon_to_hexes
from .util import PipelineError, get_logger
from .viz import plot_city_grid

log = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="heat-island",
    help="Map urban heat islands and rank tree-planting priorities for any city.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """heat-island: map urban heat islands and rank tree-planting priorities for any city.

    A callback is required here (even a no-op one) so typer always runs in "multiple
    subcommand" mode — with only one @app.command() registered it would otherwise collapse
    to a single-command CLI and `heat-island preview ...` would stop being a valid
    invocation. This is also the future home for a shared `--data-dir` option.
    """


@app.command()
def preview(
    query: str = typer.Argument(..., help="City to preview, e.g. 'Ghent, Belgium'."),
    resolution: int | None = typer.Option(
        None, "--resolution", help="H3 resolution (default: PipelineConfig default, 9)."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output PNG path (default: data/preview_<city_id>.png)."
    ),
) -> None:
    """Milestone-1 sanity check: geocode a city, build its hex grid, and plot it. No modelling."""
    cfg = PipelineConfig()
    res = resolution if resolution is not None else cfg.h3_resolution

    try:
        boundary = get_city_boundary(query, cfg)
        cells = polygon_to_hexes(boundary.geometry, res)
        hex_gdf = hexes_to_gdf(cells)
        stats = grid_stats(cells, boundary.geometry)

        out_png = out if out is not None else cfg.data_dir / f"preview_{boundary.city_id}.png"

        table = Table(title=f"{boundary.name}, {boundary.country} — {boundary.city_id} (res {res})")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("n_hexes", str(stats["n_hexes"]))
        table.add_row("area_km2", f"{stats['area_km2']:.2f}")
        table.add_row("mean_hex_area_km2", f"{stats['mean_hex_area_km2']:.5f}")
        console.print(table)

        saved_path = plot_city_grid(boundary, hex_gdf, out_png)
        console.print(f"[green]Saved plot to[/green] {saved_path}")
    except PipelineError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc


@app.command("add-city")
def add_city(
    query: str = typer.Argument(..., help="City to process, e.g. 'Berlin, Germany'."),
    resolution: int | None = typer.Option(
        None, "--resolution", help="H3 resolution (default 9; lower = coarser/faster)."
    ),
    force: bool = typer.Option(False, "--force", help="Ignore cached satellite/OSM data and refetch."),
) -> None:
    """Run the full pipeline for a city and upsert the results into data/heat.duckdb.

    Steps: boundary → H3 grid → Landsat LST → Sentinel-2 indices → DEM → OSM features →
    demographics (US only) → LightGBM model + spatial CV + SHAP → greening simulation →
    priority score → DuckDB.
    """
    from . import db
    from .features import build_feature_table
    from .model import train_and_evaluate
    from .simulate import compute_priority, greening_target_ndvi, run_greening

    cfg = PipelineConfig()
    if resolution is not None:
        cfg.h3_resolution = resolution

    try:
        console.rule(f"[bold]heat-island · {query}")
        boundary = get_city_boundary(query, cfg)
        console.print(
            f"Resolved to [bold]{boundary.name}[/bold] ({boundary.country}) — id [cyan]{boundary.city_id}[/cyan]"
        )
        if force:
            _clear_city_cache(cfg, boundary.city_id)

        df = build_feature_table(boundary, cfg)

        console.print("Training LightGBM with spatial cross-validation…")
        result = train_and_evaluate(df, cfg)

        target = greening_target_ndvi(df, cfg)
        console.print(f"Greening counterfactual: raising NDVI to {target:.3f} where below")
        df = run_greening(df, result.model, cfg)
        df = compute_priority(df)

        db.upsert_city(cfg.db_path, boundary, df, result)

        top_shap = sorted(result.shap_importance.items(), key=lambda kv: -kv[1])[:3]
        table = Table(title=f"{boundary.name} — pipeline summary")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("hexes", str(len(df)))
        table.add_row(
            "LST °C (min/mean/max)",
            f"{df.mean_lst_c.min():.1f} / {df.mean_lst_c.mean():.1f} / {df.mean_lst_c.max():.1f}",
        )
        table.add_row("spatial-CV R²", f"{result.r2:.3f}")
        table.add_row("spatial-CV MAE °C", f"{result.mae:.2f}")
        table.add_row("top SHAP features", ", ".join(f"{k} ({v:.2f})" for k, v in top_shap))
        table.add_row("max predicted cooling °C", f"{df.predicted_cooling_c.max():.2f}")
        table.add_row("hexes with priority ≥ 0.8", str(int((df.priority_score >= 0.8).sum())))
        console.print(table)
        console.print(f"[green]✓ {boundary.name} written to[/green] {cfg.db_path}")
    except PipelineError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc


@app.command("list-cities")
def list_cities() -> None:
    """List processed cities (with model metrics) from data/heat.duckdb."""
    from . import db

    cfg = PipelineConfig()
    cities = db.list_cities(cfg.db_path)
    if cities.empty:
        console.print("No cities processed yet. Run: heat-island add-city \"<City, Country>\"")
        return
    table = Table(title=f"{len(cities)} cities in {cfg.db_path}")
    for col in ("city_id", "name", "country", "n_hexes", "r2", "mae", "processed_at"):
        table.add_column(col)
    for _, row in cities.iterrows():
        table.add_row(
            str(row.get("city_id", "")),
            str(row.get("name", "")),
            str(row.get("country", ""))[:28],
            str(row.get("n_hexes", "")),
            "" if row.get("r2") is None else f"{row['r2']:.3f}",
            "" if row.get("mae") is None else f"{row['mae']:.2f}",
            str(row.get("processed_at", ""))[:19],
        )
    console.print(table)


@app.command("remove-city")
def remove_city(city_id: str = typer.Argument(..., help="city_id as shown by list-cities.")) -> None:
    """Remove a city from data/heat.duckdb (its download cache is kept for fast re-adds)."""
    from . import db

    cfg = PipelineConfig()
    n = db.remove_city(cfg.db_path, city_id)
    if n:
        console.print(f"[green]Removed[/green] {city_id} ({n} hexes).")
    else:
        console.print(f"[yellow]No city with id '{city_id}' found.[/yellow] Try: heat-island list-cities")


def _clear_city_cache(cfg: PipelineConfig, city_id: str) -> None:
    """--force: drop cached per-hex parquets + OSM HTTP cache; keep the boundary geojson."""
    import shutil

    cache = cfg.city_cache(city_id)
    removed = 0
    for f in cache.glob("*.parquet"):
        f.unlink()
        removed += 1
    if (cache / "osm_http").exists():
        shutil.rmtree(cache / "osm_http")
        removed += 1
    console.print(f"[yellow]--force: cleared {removed} cached artifacts for {city_id}[/yellow]")


if __name__ == "__main__":
    app()
