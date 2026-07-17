"""HexMap — a zoomable / pannable hex-polygon map for Dashdown.

Custom data-driven component (see ``.references/extending.md``): the Python side
only registers the tag and emits an **async placeholder** card carrying an
html-escaped JSON config. The drawing, WKT parsing, colour ramp, zoom/pan, theme
sync and filter re-fetch all live in the colocated ``HexMap.js`` (self-inits —
``app.js`` wires only the built-in async types). ``HexMap.css`` (auto-linked)
styles the card chrome + the gradient legend footer.

Tag contract::

    <HexMap data={lst_map} value="mean_lst_c" unit="°C" scheme="heat"
            title="Land-surface temperature" height=520 tooltip="ndvi,priority_score" />

The bound query (``data={query}``) must return, per hex row: ``h3``,
``geometry_wkt`` (a POLYGON / MULTIPOLYGON in EPSG:4326 lon/lat), the ``value``
column, and any optional ``tooltip`` columns.
"""

from __future__ import annotations

import html
import json
from typing import Any

from dashdown import Component, register_component

# The three named colour ramps (3 stops each, low → mid → high). Kept in sync
# with the JS SCHEMES table; only the JS actually paints, but naming them here
# documents the contract and lets us fall back to a valid default.
_SCHEMES = {"heat", "greens", "priority"}


def _query_name(data_val: Any) -> str | None:
    """The registered query name from a ``data={query}`` attr (a DataRef), or a
    bare string as a fallback. Returns None when absent."""
    name = getattr(data_val, "name", None)
    if name:
        return str(name)
    if isinstance(data_val, str) and data_val.strip():
        return data_val.strip()
    return None


@register_component("HexMap")
class HexMap(Component):
    """Zoomable hex-polygon choropleth of a per-hex metric.

    Renders each hex's true polygon (from ``geometry_wkt``) filled by a
    continuous colour ramp over ``value=``; wheel zooms, drag pans, pinch works
    on touch. Replaces the quintile-band ScatterChart pseudo-maps. Re-queries
    when the page's City filter changes (same store subscription the built-in
    charts use).
    """

    def render(self, attrs: dict[str, Any], ctx, inner: str | None = None) -> str:
        query = _query_name(attrs.get("data"))
        if not query:
            raise ValueError("HexMap requires a `data={query}` attribute")

        value = str(attrs.get("value", "") or "").strip()
        if not value:
            raise ValueError("HexMap requires a `value=` column attribute")

        scheme = str(attrs.get("scheme", "heat") or "heat").strip()
        if scheme not in _SCHEMES:
            scheme = "heat"

        unit = str(attrs.get("unit", "") or "")
        title = str(attrs.get("title", "") or "")

        # `height=520` bare-coerces to int; guard a stray string/float too.
        try:
            height = int(attrs.get("height") or 520)
        except (TypeError, ValueError):
            height = 520
        if height <= 0:
            height = 520

        # `tooltip="ndvi,priority_score"` → ["ndvi", "priority_score"].
        tooltip_raw = str(attrs.get("tooltip", "") or "")
        tooltip = [c.strip() for c in tooltip_raw.split(",") if c.strip()]

        cfg = {
            "query": query,
            "value": value,
            "unit": unit,
            "scheme": scheme,
            "title": title,
            "height": height,
            "tooltip": tooltip,
        }
        # Escaped for a quoted HTML attribute — the JS parses it back from the
        # dataset; nothing user-supplied is interpolated raw into the DOM.
        data_config = html.escape(json.dumps(cfg), quote=True)

        title_html = (
            f'<div class="hexmap-title">{html.escape(title)}</div>' if title else ""
        )
        # Card chrome mirrors the built-in chart cards (card bg-base-100 border
        # border-base-300) so a HexMap sits visually alongside them. A skeleton
        # placeholder fills the plot region until the first fetch lands.
        return (
            '<div class="hexmap card bg-base-100 border border-base-300" '
            f'data-async-component="hexmap" data-config="{data_config}" '
            'style="width:100%">'
            f"{title_html}"
            f'<div class="hexmap-region" style="height:{height}px" data-hexmap-canvas>'
            '<div class="dashdown-chart-skeleton skeleton w-full h-full"></div>'
            "</div>"
            '<div class="hexmap-legend" data-hexmap-legend></div>'
            "</div>"
        )
