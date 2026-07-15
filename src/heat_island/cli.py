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

# NOTE: add-city / list-cities / remove-city are wired in by the integrator — do not add them here.


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


if __name__ == "__main__":
    app()
