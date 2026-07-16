"""Satellite fetchers: Landsat land-surface temperature, Sentinel-2 indices, Copernicus DEM,
ESA WorldCover land cover.

They all read from Microsoft Planetary Computer (keyless, `planetary_computer.sign_inplace`),
aggregate to the city's H3 grid, and cache the result as parquet under ``cache/<city_id>/`` (the
temperature/index fetchers composite the recent summer windows; the DEM and land-cover products
are static). A cache hit skips the network entirely.

Public API (consumed by features.py):

    fetch_lst_per_hex(boundary, cfg)        -> DataFrame[h3, mean_lst_c]
    fetch_s2_indices_per_hex(boundary, cfg) -> DataFrame[h3, ndvi, ndbi, ndwi, albedo]
    fetch_elevation_per_hex(boundary, cfg)  -> DataFrame[h3, elevation]
    fetch_landcover_per_hex(boundary, cfg)  -> DataFrame[h3, plantable_fraction, tree_fraction]

Pure, unit-testable helpers (no network): ``_lst_from_dn``, ``_s2_offset_for_times``,
``_indices_from_bands``, ``_sort_and_cap_items``, ``_plantable_weights_lookup``,
``_tree_indicator``, ``_parse_major_version``, ``_latest_version_items``.

Dask note: we rely on dask's default threaded scheduler (no distributed cluster). Chunks are
2048 px so each city's spatial extent is a single chunk and task graphs stay tiny; the only
parallelism is over the modest number of scenes. This keeps CPU/memory friendly when several
city pipelines run concurrently on the same box.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .boundary import CityBoundary
from .config import PipelineConfig
from .hexgrid import points_to_hex_means
from .util import DataUnavailableError, get_logger, retry_call

log = get_logger(__name__)

# --- constants ---------------------------------------------------------------

# Landsat Collection-2 Level-2 surface-temperature scaling (ST_B10 / lwir11 DN -> Kelvin -> °C).
_LST_SCALE = 0.00341802
_LST_OFFSET_K = 149.0
_KELVIN = 273.15
# QA_PIXEL bits 0-4 = fill, dilated cloud, cirrus, cloud, cloud shadow. Any set -> drop pixel.
_QA_MASK = 0b11111
# Plausible surface-temperature envelope (°C); anything outside is a masking/scaling artefact.
_LST_MIN_C, _LST_MAX_C = -20.0, 70.0
# Milestone gate: urban summer LST typically 25-50 °C (a WARNING, not an error — cool-climate
# cities can sit lower).
_LST_GATE_LO, _LST_GATE_HI = 25.0, 50.0

# Sentinel-2 L2A "harmonize" offset: from processing baseline 04.00 (rolled out 2022-01-25) the
# products carry a +1000 DN radiometric offset, removed via (DN - 1000). We key the decision on
# the acquisition DATE rather than per-item ``s2:processing_baseline`` because grouping by
# solar_day collapses several items into one time slice, so a per-item property no longer maps
# cleanly onto the composited time axis; every PC L2A product on/after the cutoff uses the new
# baseline, making the date rule both reliable and simpler. (ARCHITECTURE.md sanctions this.)
_S2_BASELINE_CUTOFF = np.datetime64("2022-01-25")
_S2_OFFSET_NEW = -1000.0
# SCL classes to drop: 0 no-data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium prob,
# 9 cloud high prob, 10 thin cirrus.
_SCL_DROP = [0, 1, 3, 8, 9, 10]
_S2_NDVI_GATE = (-0.1, 0.9)

# ESA WorldCover land cover: a static, global, categorical 10 m map (uint8 class codes 10..100,
# nodata 0 per the asset's raster:bands metadata). Items carry no single ``datetime`` (they use a
# year-long start/end range), so we select the newest product by ``esa_worldcover:product_version``
# ("2.0.0" = 2021 map, "1.0.0" = 2020) rather than by time. Probed live on Planetary Computer.
_LC_COLLECTION = "esa-worldcover"
_LC_MAP_ASSET_KEYS = ("map",)  # categorical "Land Cover Classes" asset
_LC_VERSION_PROP = "esa_worldcover:product_version"
_LC_NODATA = 0

_DENOM_EPS = 1e-6  # guard band for normalized-difference denominators

_CHUNKS = {"x": 2048, "y": 2048}

# --- lazy module-level catalog cache -----------------------------------------

_CATALOG: Any = None


def _open_catalog() -> Any:
    """Open (once) the Planetary Computer STAC catalog with request signing."""
    global _CATALOG
    if _CATALOG is None:
        import planetary_computer
        import pystac_client

        _CATALOG = pystac_client.Client.open(
            config.STAC_URL, modifier=planetary_computer.sign_inplace
        )
        log.info("opened STAC catalog %s", config.STAC_URL)
    return _CATALOG


# --- pure helpers (unit-tested offline) --------------------------------------


def _lst_from_dn(dn: xr.DataArray, qa: xr.DataArray) -> xr.DataArray:
    """Masked land-surface temperature in °C from raw ST DN + QA_PIXEL.

    Drops nodata (DN == 0) and any pixel with a QA bit 0-4 set, applies the Collection-2 ST
    scaling, then nulls values outside the plausible [-20, 70] °C envelope. Broadcasts over any
    leading dims (e.g. time), so it can be applied to the whole stack before compositing.
    """
    valid = (dn != 0) & ((qa & _QA_MASK) == 0)
    lst_c = dn.where(valid) * _LST_SCALE + _LST_OFFSET_K - _KELVIN
    return lst_c.where((lst_c >= _LST_MIN_C) & (lst_c <= _LST_MAX_C))


def _s2_offset_for_times(times: Any) -> np.ndarray:
    """Per-timestamp Sentinel-2 harmonize offset (-1000 on/after the 04.00 baseline, else 0).

    ``times`` is any array-like of datetime64 (e.g. ``ds.time.values``). Returns a float ndarray
    aligned to ``times`` — wrap it in a ``dims=("time",)`` DataArray to broadcast over the stack.
    """
    dates = pd.to_datetime(np.asarray(times)).to_numpy(dtype="datetime64[ns]")
    return np.where(dates >= _S2_BASELINE_CUTOFF, _S2_OFFSET_NEW, 0.0)


def _safe_ratio(num: Any, den: Any) -> Any:
    """``num / den`` with |den| < eps mapped to NaN. Works on numpy arrays and xarray DataArrays.

    ``xr.where`` preserves DataArray coordinates (needed to keep lat/lon for hex aggregation) and
    degrades to a plain ndarray for numpy inputs, so the same helper serves the pipeline and the
    offline unit tests.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    return xr.where(np.abs(den) < _DENOM_EPS, np.nan, ratio)


def _indices_from_bands(
    b02: Any, b03: Any, b04: Any, b08: Any, b11: Any
) -> tuple[Any, Any, Any, Any]:
    """NDVI, NDBI, NDWI and a crude visible-band albedo proxy from surface-reflectance bands.

    ndvi=(B08-B04)/(B08+B04), ndbi=(B11-B08)/(B11+B08), ndwi=(B03-B08)/(B03+B08),
    albedo=(B02+B03+B04)/3 (documented crude proxy). Zero denominators -> NaN.
    """
    ndvi = _safe_ratio(b08 - b04, b08 + b04)
    ndbi = _safe_ratio(b11 - b08, b11 + b08)
    ndwi = _safe_ratio(b03 - b08, b03 + b08)
    albedo = (b02 + b03 + b04) / 3.0
    return ndvi, ndbi, ndwi, albedo


def _plantable_weights_lookup(classes: Any) -> Any:
    """ESA WorldCover class codes -> per-pixel plantable weight in [0, 1] (vectorized).

    Weights come from ``config.PLANTABLE_CLASS_WEIGHTS``; a class absent from that table maps to
    0.0 and nodata (class ``_LC_NODATA`` == 0) maps to NaN. Built as a sum of mutually exclusive
    masked class indicators so it stays lazy on a dask/xarray stack (preserving lat/lon/time
    coords for the median-composite downstream) yet also works on a plain numpy array offline.
    """
    weights = sum(float(w) * (classes == cls) for cls, w in config.PLANTABLE_CLASS_WEIGHTS.items())
    return xr.where(classes == _LC_NODATA, np.nan, weights)


def _tree_indicator(classes: Any) -> Any:
    """ESA WorldCover tree-cover indicator: 1.0 where class == ``LANDCOVER_TREE_CLASS`` else 0.0.

    Nodata (class ``_LC_NODATA``) maps to NaN. Same lazy/numpy duality (and coord preservation on
    an xarray input) as ``_plantable_weights_lookup``.
    """
    tree = (classes == config.LANDCOVER_TREE_CLASS).astype("float64")
    return xr.where(classes == _LC_NODATA, np.nan, tree)


def _sort_and_cap_items(items: Sequence[Any], max_items: int) -> list[Any]:
    """Sort STAC items by ``eo:cloud_cover`` ascending (missing -> 100) and cap at ``max_items``."""

    def cloud(item: Any) -> float:
        val = item.properties.get("eo:cloud_cover")
        return float(val) if val is not None else 100.0

    return sorted(items, key=cloud)[:max_items]


def _parse_major_version(version: Any) -> int | None:
    """Major integer of a dotted version string (``"2.0.0" -> 2``); None if absent/unparseable.

    Comparing the parsed major int avoids the pitfalls of string-comparing versions (``"10" <
    "9"`` lexically), which is all WorldCover needs to rank ``"2.0.0"`` above ``"1.0.0"``.
    """
    if version is None:
        return None
    try:
        return int(str(version).split(".")[0])
    except (ValueError, IndexError):
        return None


def _item_time(item: Any) -> str:
    """Sortable acquisition time used only as a version tie-breaker: the item ``datetime`` if set,
    else its ``start_datetime`` (WorldCover items carry no single ``datetime``), else ``""``."""
    dt = getattr(item, "datetime", None)
    if dt is not None:
        return dt.isoformat()
    return str(item.properties.get("start_datetime", ""))


def _latest_version_items(items: Sequence[Any]) -> list[Any]:
    """Keep only the ESA WorldCover items of the newest product version.

    Items are ranked by the parsed major of ``esa_worldcover:product_version``; if no item carries
    that property we fall back to keeping the items of the latest ``_item_time``.
    """
    majors = [_parse_major_version(it.properties.get(_LC_VERSION_PROP)) for it in items]
    if any(m is not None for m in majors):
        top = max(m for m in majors if m is not None)
        kept = [it for it, m in zip(items, majors) if m == top]
        log.info("land cover: %d/%d item(s) at product version major %d", len(kept), len(items), top)
        return kept

    latest = max((_item_time(it) for it in items), default="")
    kept = [it for it in items if _item_time(it) == latest]
    log.info(
        "land cover: no version property — kept %d/%d item(s) at latest time %s",
        len(kept), len(items), latest or "?",
    )
    return kept


# --- search / geometry / IO internals ----------------------------------------


def _bbox(boundary: CityBoundary, cfg: PipelineConfig) -> tuple[float, float, float, float]:
    """City bounds (EPSG:4326) padded by ``cfg.bbox_pad_deg`` on every side."""
    minx, miny, maxx, maxy = boundary.geometry.bounds
    p = cfg.bbox_pad_deg
    return (minx - p, miny - p, maxx + p, maxy + p)


def _windows(boundary: CityBoundary, cfg: PipelineConfig) -> list[tuple[str, str]]:
    from .util import summer_windows

    return summer_windows(boundary.centroid_lat, years_back=cfg.years_back)


def _scene_dates(items: Iterable[Any]) -> list:
    return sorted(it.datetime for it in items if getattr(it, "datetime", None) is not None)


def _search_items(
    catalog: Any,
    collections: Sequence[str],
    bbox: Sequence[float],
    windows: Sequence[tuple[str, str]],
    build_query: Callable[[int], dict],
    max_items: int,
    cfg: PipelineConfig,
    what: str,
) -> list[Any]:
    """Search each summer window, merge, sort by cloud, cap. Relax cloud once, else raise."""
    cloud_levels = [cfg.max_cloud_pct]
    if cfg.relaxed_cloud_pct != cfg.max_cloud_pct:
        cloud_levels.append(cfg.relaxed_cloud_pct)

    for attempt, cloud in enumerate(cloud_levels):
        items: list[Any] = []
        for start, end in windows:
            search = catalog.search(
                collections=list(collections),
                bbox=list(bbox),
                datetime=f"{start}/{end}",
                query=build_query(cloud),
            )
            got = retry_call(
                lambda s=search: list(s.item_collection()),
                what=f"{what} search {start[:7]} <{cloud}% cloud",
            )
            items.extend(got)

        capped = _sort_and_cap_items(items, max_items)
        if capped:
            if attempt > 0:
                log.warning(
                    "%s: only found scenes after relaxing cloud cover to <%d%%", what, cloud
                )
            log.info(
                "%s: %d candidate scenes across %d windows, using %d (<%d%% cloud)",
                what, len(items), len(windows), len(capped), cloud,
            )
            return capped

        log.warning(
            "%s: no scenes <%d%% cloud across windows %s%s",
            what, cloud, [w[0][:7] for w in windows],
            " — relaxing once" if attempt == 0 and len(cloud_levels) > 1 else "",
        )

    raise DataUnavailableError(
        f"No {list(collections)} scenes found over "
        f"{[f'{s}/{e}' for s, e in windows]} even at <{cloud_levels[-1]}% cloud. "
        "The area may be persistently cloudy or outside coverage."
    )


def _dataset_to_points(ds: xr.Dataset, value_cols: list[str]) -> pd.DataFrame:
    """Flatten a lat/lon-gridded Dataset to a tidy DataFrame[lat, lon, *value_cols].

    Rows where every value column is NaN are dropped (no signal to aggregate); a NaN in only
    some columns is preserved so ``points_to_hex_means`` can skipna per column.
    """
    df = ds[value_cols].to_dataframe().reset_index()
    df = df.rename(columns={"latitude": "lat", "longitude": "lon", "y": "lat", "x": "lon"})
    df = df[["lat", "lon", *value_cols]]
    return df.dropna(subset=value_cols, how="all")


def _grid_to_points(da_2d: xr.DataArray, value_name: str) -> pd.DataFrame:
    """A single 2D (lat/lon) DataArray -> DataFrame[lat, lon, value_name] with NaN rows dropped."""
    return _dataset_to_points(da_2d.to_dataset(name=value_name), [value_name])


def _drop_time(da: xr.DataArray) -> xr.DataArray:
    """Collapse a possible ``time`` dim: median over multiple slices, squeeze a single one."""
    if "time" not in da.dims:
        return da
    if da.sizes["time"] > 1:
        return da.median(dim="time", skipna=True)
    return da.squeeze("time", drop=True)


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.nanmin(values)),
        float(np.nanmean(values)),
        float(np.nanmax(values)),
    )


def _cache_path(cfg: PipelineConfig, city_id: str, kind: str):
    return cfg.city_cache(city_id) / f"{kind}_res{cfg.h3_resolution}.parquet"


def _read_cache(path, what: str) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_parquet(path)
        log.info("%s cache hit: %s (%d hexes)", what, path, len(df))
        return df
    return None


def _write_cache(df: pd.DataFrame, path) -> None:
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d hexes)", path, len(df))


def _check_unit_interval(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    """WARN if any aggregated fraction escapes [0, 1]. The class weights are bounded to [0, 1] and
    hex means of bounded values stay bounded, so this never fires in practice — it guards against a
    reclassification/compositing regression rather than silently clipping."""
    for col in cols:
        vals = df[col].to_numpy()
        finite = vals[np.isfinite(vals)]
        if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            log.warning(
                "land cover %s: %s outside [0, 1] (min %.3f, max %.3f) — check reclassification",
                name, col, float(finite.min()), float(finite.max()),
            )


# --- public fetchers ---------------------------------------------------------


def fetch_lst_per_hex(boundary: CityBoundary, cfg: PipelineConfig) -> pd.DataFrame:
    """Median summer land-surface temperature per H3 hex (°C). Columns: h3, mean_lst_c."""
    cache = _cache_path(cfg, boundary.city_id, "lst")
    cached = _read_cache(cache, "LST")
    if cached is not None:
        return cached

    import odc.stac

    catalog = _open_catalog()
    bbox = _bbox(boundary, cfg)
    windows = _windows(boundary, cfg)

    items = _search_items(
        catalog,
        ["landsat-c2-l2"],
        bbox,
        windows,
        build_query=lambda cloud: {
            "eo:cloud_cover": {"lt": cloud},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
        max_items=cfg.landsat_max_scenes,
        cfg=cfg,
        what="Landsat LST",
    )

    assets = items[0].assets
    st_key = next((k for k in ("lwir11", "ST_B10", "lwir") if k in assets), None)
    qa_key = next((k for k in ("qa_pixel", "QA_PIXEL") if k in assets), None)
    if st_key is None or qa_key is None:
        raise DataUnavailableError(
            "landsat-c2-l2 item is missing a surface-temperature and/or QA asset "
            f"(have: {sorted(assets)}). Expected one of lwir11/ST_B10/lwir + qa_pixel/QA_PIXEL."
        )

    ds = odc.stac.load(
        items,
        bands=[st_key, qa_key],
        groupby="solar_day",
        crs="EPSG:4326",
        resolution=cfg.raster_res_deg,
        bbox=list(bbox),
        chunks=_CHUNKS,
    )

    lst_c = _lst_from_dn(ds[st_key], ds[qa_key])
    composite = lst_c.median(dim="time", skipna=True)
    composite = retry_call(composite.compute, what="Landsat LST composite compute")

    values = composite.values
    if not np.isfinite(values).any():
        raise DataUnavailableError(
            f"Landsat LST composite for '{boundary.name}' is entirely masked — every pixel was "
            "cloud/QA-flagged or out of range across all scenes."
        )

    dates = _scene_dates(items)
    mn, mean, mx = _stats(values)
    n_time = composite.sizes.get("time", ds.sizes.get("time", len(items)))
    log.info(
        "LST %s: %d scenes (%s..%s) -> %d solar-day composites | %.1f/%.1f/%.1f °C (min/mean/max)",
        boundary.name, len(items),
        dates[0].date() if dates else "?", dates[-1].date() if dates else "?",
        int(n_time), mn, mean, mx,
    )
    if not (_LST_GATE_LO <= mean <= _LST_GATE_HI):
        log.warning(
            "LST %s: mean composite %.1f °C is outside the expected urban-summer band "
            "%.0f-%.0f °C — check scenes/masking (cool-climate cities may legitimately be lower)",
            boundary.name, mean, _LST_GATE_LO, _LST_GATE_HI,
        )

    points = _grid_to_points(composite, "mean_lst_c")
    hexdf = points_to_hex_means(points, cfg.h3_resolution, ["mean_lst_c"])
    _write_cache(hexdf, cache)
    return hexdf


def fetch_s2_indices_per_hex(boundary: CityBoundary, cfg: PipelineConfig) -> pd.DataFrame:
    """Median summer Sentinel-2 indices per H3 hex. Columns: h3, ndvi, ndbi, ndwi, albedo."""
    cache = _cache_path(cfg, boundary.city_id, "s2")
    cached = _read_cache(cache, "S2")
    if cached is not None:
        return cached

    import odc.stac

    catalog = _open_catalog()
    bbox = _bbox(boundary, cfg)
    windows = _windows(boundary, cfg)

    bands = ["B02", "B03", "B04", "B08", "B11", "SCL"]
    items = _search_items(
        catalog,
        ["sentinel-2-l2a"],
        bbox,
        windows,
        build_query=lambda cloud: {"eo:cloud_cover": {"lt": cloud}},
        max_items=cfg.s2_max_scenes,
        cfg=cfg,
        what="Sentinel-2",
    )

    ds = odc.stac.load(
        items,
        bands=bands,
        groupby="solar_day",
        crs="EPSG:4326",
        resolution=cfg.s2_res_deg,
        bbox=list(bbox),
        chunks=_CHUNKS,
    )

    # Per-time harmonize offset broadcast over the stack (see _S2_BASELINE_CUTOFF note).
    offset = xr.DataArray(
        _s2_offset_for_times(ds["time"].values), dims=("time",), coords={"time": ds["time"]}
    )
    scl_ok = ~ds["SCL"].isin(_SCL_DROP)

    medians = {}
    for band in ("B02", "B03", "B04", "B08", "B11"):
        reflectance = ((ds[band] + offset) / 10000.0).where(scl_ok).clip(0.0, 1.0)
        medians[band] = reflectance.median(dim="time", skipna=True)

    composite = retry_call(xr.Dataset(medians).compute, what="Sentinel-2 composite compute")

    ndvi, ndbi, ndwi, albedo = _indices_from_bands(
        composite["B02"], composite["B03"], composite["B04"], composite["B08"], composite["B11"]
    )
    idx = xr.Dataset({"ndvi": ndvi, "ndbi": ndbi, "ndwi": ndwi, "albedo": albedo})

    if not np.isfinite(idx["ndvi"].values).any():
        raise DataUnavailableError(
            f"Sentinel-2 index composite for '{boundary.name}' is entirely masked — every pixel "
            "was SCL-flagged across all scenes."
        )

    dates = _scene_dates(items)
    ndvi_mean = float(np.nanmean(idx["ndvi"].values))
    log.info(
        "S2 %s: %d scenes (%s..%s) | mean NDVI %.2f NDBI %.2f NDWI %.2f albedo %.2f",
        boundary.name, len(items),
        dates[0].date() if dates else "?", dates[-1].date() if dates else "?",
        ndvi_mean,
        float(np.nanmean(idx["ndbi"].values)),
        float(np.nanmean(idx["ndwi"].values)),
        float(np.nanmean(idx["albedo"].values)),
    )
    if not (_S2_NDVI_GATE[0] <= ndvi_mean <= _S2_NDVI_GATE[1]):
        log.warning(
            "S2 %s: mean NDVI %.2f is outside the expected [%.1f, %.1f] range — check masking",
            boundary.name, ndvi_mean, *_S2_NDVI_GATE,
        )

    value_cols = ["ndvi", "ndbi", "ndwi", "albedo"]
    points = _dataset_to_points(idx, value_cols)
    hexdf = points_to_hex_means(points, cfg.h3_resolution, value_cols)
    _write_cache(hexdf, cache)
    return hexdf


def fetch_elevation_per_hex(boundary: CityBoundary, cfg: PipelineConfig) -> pd.DataFrame:
    """Mean Copernicus DEM elevation per H3 hex (metres). Columns: h3, elevation."""
    cache = _cache_path(cfg, boundary.city_id, "dem")
    cached = _read_cache(cache, "DEM")
    if cached is not None:
        return cached

    import odc.stac

    catalog = _open_catalog()
    bbox = _bbox(boundary, cfg)

    # DEM is static: no datetime, no cloud filter.
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=list(bbox))
    items = retry_call(
        lambda: list(search.item_collection()), what="cop-dem-glo-30 search"
    )
    if not items:
        raise DataUnavailableError(
            f"cop-dem-glo-30 returned no tiles for '{boundary.name}' (bbox {tuple(round(b, 3) for b in bbox)})."
        )
    log.info("DEM %s: %d tile(s) intersect the bbox", boundary.name, len(items))

    ds = odc.stac.load(
        items,
        bands=["data"],
        groupby="solar_day",
        crs="EPSG:4326",
        resolution=cfg.raster_res_deg,
        bbox=list(bbox),
        chunks=_CHUNKS,
    )

    elevation = _drop_time(ds["data"])
    elevation = retry_call(elevation.compute, what="cop-dem-glo-30 compute")

    values = elevation.values
    if not np.isfinite(values).any():
        raise DataUnavailableError(
            f"Copernicus DEM tile for '{boundary.name}' is entirely nodata over the bbox."
        )
    mn, mean, mx = _stats(values)
    log.info("DEM %s: elevation %.1f/%.1f/%.1f m (min/mean/max)", boundary.name, mn, mean, mx)

    points = _grid_to_points(elevation, "elevation")
    hexdf = points_to_hex_means(points, cfg.h3_resolution, ["elevation"])
    _write_cache(hexdf, cache)
    return hexdf


def fetch_landcover_per_hex(boundary: CityBoundary, cfg: PipelineConfig) -> pd.DataFrame:
    """Plantable and tree-cover fractions per H3 hex from ESA WorldCover. Columns: h3,
    plantable_fraction, tree_fraction (both in [0, 1]).

    The static 10 m categorical map is reclassified per pixel to a plantable weight
    (``config.PLANTABLE_CLASS_WEIGHTS``) and a tree indicator **before** any compositing — never
    median raw class codes — then median-composited over tiles and averaged per hex.
    """
    cache = _cache_path(cfg, boundary.city_id, "landcover")
    cached = _read_cache(cache, "land cover")
    if cached is not None:
        return cached

    import odc.stac

    catalog = _open_catalog()
    bbox = _bbox(boundary, cfg)

    # Static product: search without datetime / cloud filter, then keep the newest version only.
    search = catalog.search(collections=[_LC_COLLECTION], bbox=list(bbox))
    items = retry_call(lambda: list(search.item_collection()), what="esa-worldcover search")
    if not items:
        raise DataUnavailableError(
            f"{_LC_COLLECTION} returned no tiles for '{boundary.name}' "
            f"(bbox {tuple(round(b, 3) for b in bbox)})."
        )
    items = _latest_version_items(items)

    assets = items[0].assets
    map_key = next((k for k in _LC_MAP_ASSET_KEYS if k in assets), None)
    if map_key is None:
        raise DataUnavailableError(
            f"{_LC_COLLECTION} item is missing the land-cover map asset "
            f"(have: {sorted(assets)}). Expected one of {list(_LC_MAP_ASSET_KEYS)}."
        )

    ds = odc.stac.load(
        items,
        bands=[map_key],
        groupby="solar_day",
        crs="EPSG:4326",
        resolution=cfg.landcover_res_deg,
        bbox=list(bbox),
        chunks=_CHUNKS,
    )

    # Reclassify categorical codes -> plantable weight + tree indicator per slice, THEN composite.
    classes = ds[map_key]
    plantable = _drop_time(_plantable_weights_lookup(classes))
    tree = _drop_time(_tree_indicator(classes))
    composite = xr.Dataset({"plantable_fraction": plantable, "tree_fraction": tree})
    composite = retry_call(composite.compute, what="ESA WorldCover composite compute")

    if not np.isfinite(composite["plantable_fraction"].values).any():
        raise DataUnavailableError(
            f"ESA WorldCover map for '{boundary.name}' is entirely nodata over the bbox."
        )

    value_cols = ["plantable_fraction", "tree_fraction"]
    points = _dataset_to_points(composite, value_cols)
    hexdf = points_to_hex_means(points, cfg.h3_resolution, value_cols)
    _check_unit_interval(hexdf, value_cols, boundary.name)

    version = items[0].properties.get(_LC_VERSION_PROP, "?")
    p_mn, p_mean, p_mx = _stats(hexdf["plantable_fraction"].to_numpy())
    log.info(
        "land cover %s: %d tile(s) v%s -> %d hexes | plantable %.2f/%.2f/%.2f (min/mean/max) "
        "| tree %.2f (mean)",
        boundary.name, len(items), version, len(hexdf),
        p_mn, p_mean, p_mx, float(np.nanmean(hexdf["tree_fraction"].to_numpy())),
    )

    _write_cache(hexdf, cache)
    return hexdf
