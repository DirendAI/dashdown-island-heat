"""Shared utilities: logging, errors, retries, slugs, UTM helpers, summer windows."""

from __future__ import annotations

import datetime as dt
import logging
import re
import unicodedata
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

console = Console(stderr=True)

_LOGGING_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, show_path=False, markup=False)],
        )
        # quiet the noisy stacks
        for noisy in ("botocore", "urllib3", "rasterio", "distributed", "azure",
                      "pystac_client", "matplotlib", "fiona", "requests"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _LOGGING_CONFIGURED = True
    return logging.getLogger(name)


class PipelineError(RuntimeError):
    """Actionable, user-facing pipeline failure."""


class CityNotFoundError(PipelineError):
    pass


class DataUnavailableError(PipelineError):
    pass


T = TypeVar("T")


def retry_call(fn: Callable[..., T], *args: Any, what: str = "network call", **kwargs: Any) -> T:
    """Run fn(*args, **kwargs) with 4 attempts and exponential backoff (2s, 4s, 8s)."""
    log = get_logger(__name__)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        retry=retry_if_exception_type(Exception),
        reraise=True,
        before_sleep=lambda rs: log.warning(
            "%s failed (attempt %d/4): %s — retrying", what, rs.attempt_number, rs.outcome.exception()
        ),
    )
    def _inner() -> T:
        return fn(*args, **kwargs)

    return _inner()


def slugify(text: str) -> str:
    """ASCII, lowercase, hyphen-separated. 'Gent / Belgïe' -> 'gent-belgie'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return re.sub(r"-{2,}", "-", text) or "unnamed"


def utm_crs_for(obj: Any):
    """Estimated UTM CRS for a GeoDataFrame/GeoSeries or a shapely geometry in EPSG:4326."""
    import geopandas as gpd

    if hasattr(obj, "estimate_utm_crs"):
        return obj.estimate_utm_crs()
    return gpd.GeoSeries([obj], crs="EPSG:4326").estimate_utm_crs()


def summer_windows(lat: float, today: dt.date | None = None, years_back: int = 3) -> list[tuple[str, str]]:
    """The `years_back` most recent *completed* summer windows for the hemisphere at `lat`.

    Northern hemisphere: Jun 1 – Aug 31.  Southern: Dec 1 – Feb 28/29 (spans new year).
    Tropics (|lat| < 10): calendar years (no meaningful thermal summer).
    Returns ISO "YYYY-MM-DD/YYYY-MM-DD" pairs as (start, end) tuples, most recent first.
    """
    today = today or dt.date.today()
    windows: list[tuple[str, str]] = []
    if abs(lat) < 10:
        # whole calendar years, most recent completed first
        last = today.year - 1
        for y in range(last, last - years_back, -1):
            windows.append((f"{y}-01-01", f"{y}-12-31"))
        return windows
    if lat >= 10:  # northern
        last = today.year if today >= dt.date(today.year, 8, 31) else today.year - 1
        for y in range(last, last - years_back, -1):
            windows.append((f"{y}-06-01", f"{y}-08-31"))
        return windows
    # southern: summer labelled by its ending year (Dec y-1 → Feb y); leap-aware completeness
    feb_end_now = 29 if (today.year % 4 == 0 and (today.year % 100 != 0 or today.year % 400 == 0)) else 28
    last_end = today.year if today >= dt.date(today.year, 2, feb_end_now) else today.year - 1
    for y in range(last_end, last_end - years_back, -1):
        feb_end = 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
        windows.append((f"{y - 1}-12-01", f"{y}-02-{feb_end:02d}"))
    return windows
