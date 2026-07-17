// HexMap — zoomable / pannable hex-polygon map (custom Dashdown component).
//
// Self-inits (app.js wires only the built-in async types). Draws each hex's true
// polygon from `geometry_wkt` as an ECharts `custom` series, filled by a
// continuous colour ramp over the `value` column. Wheel zooms, drag pans, pinch
// works on touch (two `inside` dataZooms). Re-fetches on a filter change by
// subscribing to the same `Alpine.store("filters")` the built-in charts use, and
// re-themes on the same `onThemeChange` signal chart.js listens to.
//
// Uses the page-global `echarts` (vendored <script> in page.html) — not imported.

import { fetchQueryData, recordsOf, esc } from "dashdown/core.js";
import {
  currentEChartsTheme,
  onThemeChange,
} from "dashdown/components/echarts_theme.js";

"use strict";

// 3-stop colour ramps (low → mid → high). Kept in sync with HexMap.py's _SCHEMES.
const SCHEMES = {
  heat: ["#4c78a8", "#f2cf5b", "#e45756"],
  greens: ["#a6611a", "#c2e699", "#238443"],
  priority: ["#4c78a8", "#f58518", "#e45756"],
};

const DEG2RAD = Math.PI / 180;

/* ----------------------------- colour helpers ---------------------------- */

function hexToRgb(hex) {
  const h = String(hex).replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

/** Linear interpolation across an N-stop ramp; `t` clamped to [0,1]. */
function rampColor(stopsRgb, t) {
  if (!Number.isFinite(t)) t = 0;
  t = Math.max(0, Math.min(1, t));
  const segs = stopsRgb.length - 1;
  const scaled = t * segs;
  const i = Math.min(Math.floor(scaled), segs - 1);
  const f = scaled - i;
  const a = stopsRgb[i];
  const b = stopsRgb[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

function themeStroke(theme) {
  return theme === "dashdown-dark"
    ? "rgba(226, 232, 240, 0.16)"
    : "rgba(15, 23, 42, 0.14)";
}

function themeMuted(theme) {
  // Grey for a hex whose value is NaN/missing.
  return theme === "dashdown-dark" ? "#475569" : "#cbd5e1";
}

/* ------------------------------- WKT parsing ----------------------------- */

/**
 * Outer ring of a WKT POLYGON / MULTIPOLYGON as [[lon,lat], ...]. Robust to
 * extra whitespace; returns null if it can't parse ≥3 vertices. For a
 * MULTIPOLYGON only the first polygon's outer ring is used; polygon holes are
 * ignored (the capture stops at the first ring's closing paren).
 */
function parseWktRing(wkt) {
  if (typeof wkt !== "string") return null;
  const s = wkt.trim();
  const up = s.toUpperCase();
  let m;
  if (up.startsWith("MULTIPOLYGON")) {
    m = s.match(/\(\s*\(\s*\(([^()]*)\)/);
  } else if (up.startsWith("POLYGON")) {
    m = s.match(/\(\s*\(([^()]*)\)/);
  } else {
    return null;
  }
  if (!m) return null;
  const ring = [];
  for (const pair of m[1].split(",")) {
    const parts = pair.trim().split(/\s+/);
    if (parts.length < 2) continue;
    const x = parseFloat(parts[0]);
    const y = parseFloat(parts[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) ring.push([x, y]);
  }
  return ring.length >= 3 ? ring : null;
}

/* ------------------------------ data prepare ----------------------------- */

/**
 * Project rows to an equal-aspect plane (x = lon·cos(meanLat), y = lat) and
 * collect: per-hex shapes (projected ring or a fallback dot), the padded data
 * extent, and the value min/max. Rows with no parseable geometry fall back to a
 * dot at (lon,lat) when those columns exist, else are skipped.
 */
function prepare(records, cfg) {
  const valueCol = cfg.value;

  // Pass 1 — parse rings and accumulate a mean latitude for the projection.
  const parsed = [];
  let latSum = 0;
  let latN = 0;
  for (const rec of records) {
    const ring = parseWktRing(rec.geometry_wkt);
    let cLat = null;
    if (ring) {
      let s = 0;
      for (const p of ring) s += p[1];
      cLat = s / ring.length;
    } else if (Number.isFinite(Number(rec.lat))) {
      cLat = Number(rec.lat);
    }
    if (cLat != null) {
      latSum += cLat;
      latN += 1;
    }
    parsed.push({ rec, ring, cLat });
  }
  const meanLat = latN ? latSum / latN : 0;
  const cosLat = Math.cos(meanLat * DEG2RAD) || 1;

  // Pass 2 — project, build shapes, track extents + value range.
  const shapes = [];
  let vmin = Infinity;
  let vmax = -Infinity;
  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;
  // Raw (unprojected) lon extent — kept for introspection/verification only.
  let lonMin = Infinity;
  let lonMax = -Infinity;

  const grow = (px, py) => {
    if (px < xmin) xmin = px;
    if (px > xmax) xmax = px;
    if (py < ymin) ymin = py;
    if (py > ymax) ymax = py;
  };

  for (const { rec, ring } of parsed) {
    const raw = Number(rec[valueCol]);
    const v = Number.isFinite(raw) ? raw : NaN;
    if (Number.isFinite(v)) {
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
    if (ring) {
      const proj = ring.map(([lon, lat]) => [lon * cosLat, lat]);
      for (const p of proj) grow(p[0], p[1]);
      for (const [lon] of ring) {
        if (lon < lonMin) lonMin = lon;
        if (lon > lonMax) lonMax = lon;
      }
      // Centroid carried as the item's coord (tooltip anchor / hit region).
      let cx = 0;
      let cy = 0;
      for (const p of proj) {
        cx += p[0];
        cy += p[1];
      }
      shapes.push({
        kind: "poly",
        proj,
        c: [cx / proj.length, cy / proj.length],
        v,
        rec,
      });
    } else if (
      Number.isFinite(Number(rec.lon)) &&
      Number.isFinite(Number(rec.lat))
    ) {
      const lon = Number(rec.lon);
      const px = lon * cosLat;
      const py = Number(rec.lat);
      grow(px, py);
      if (lon < lonMin) lonMin = lon;
      if (lon > lonMax) lonMax = lon;
      shapes.push({ kind: "dot", proj: [[px, py]], c: [px, py], v, rec });
    }
    // else: unparseable geometry and no lon/lat → skip the row.
  }

  if (!Number.isFinite(vmin)) {
    vmin = 0;
    vmax = 1;
  }
  if (vmin === vmax) vmax = vmin + 1; // avoid divide-by-zero in the ramp
  if (!Number.isFinite(xmin)) {
    xmin = 0;
    xmax = 1;
    ymin = 0;
    ymax = 1;
  }

  // ~2% padding around the extent so edge hexes aren't clipped by the frame.
  const padX = (xmax - xmin) * 0.02 || 0.001;
  const padY = (ymax - ymin) * 0.02 || 0.001;
  const ext = {
    xmin: xmin - padX,
    xmax: xmax + padX,
    ymin: ymin - padY,
    ymax: ymax + padY,
  };
  if (!Number.isFinite(lonMin)) {
    lonMin = 0;
    lonMax = 0;
  }
  return { shapes, ext, vmin, vmax, lonMin, lonMax, count: shapes.length };
}

/**
 * A grid rectangle (px) inside the region that has the *data* aspect ratio and
 * is centred — so the projected city keeps its true shape regardless of the
 * card's own aspect (letterboxed rather than stretched to fill).
 */
function gridRect(prep, w, h) {
  const dataW = prep.ext.xmax - prep.ext.xmin;
  const dataH = prep.ext.ymax - prep.ext.ymin;
  const pad = 6;
  const availW = Math.max(1, w - pad * 2);
  const availH = Math.max(1, h - pad * 2);
  if (dataW <= 0 || dataH <= 0) {
    return { left: pad, top: pad, width: availW, height: availH };
  }
  const dataAspect = dataW / dataH;
  let gw;
  let gh;
  if (availW / availH > dataAspect) {
    gh = availH;
    gw = availH * dataAspect;
  } else {
    gw = availW;
    gh = availW / dataAspect;
  }
  return {
    left: (w - gw) / 2,
    top: (h - gh) / 2,
    width: gw,
    height: gh,
  };
}

/* ------------------------------ ECharts option --------------------------- */

function tooltipHtml(shape, cfg) {
  if (!shape) return "";
  const rec = shape.rec || {};
  const out = [];
  if (rec.h3 != null) {
    out.push(`<div class="hexmap-tt-id">${esc(String(rec.h3))}</div>`);
  }
  const vtxt = Number.isFinite(shape.v) ? shape.v.toFixed(2) : "n/a";
  out.push(
    `<div>${esc(cfg.value)}: <b>${esc(vtxt)}${esc(cfg.unit || "")}</b></div>`
  );
  for (const col of cfg.tooltip || []) {
    if (col === cfg.value || col === "h3" || col === "geometry_wkt") continue;
    const raw = rec[col];
    if (raw == null || raw === "") continue;
    const num = Number(raw);
    const disp = Number.isFinite(num) ? num.toFixed(2) : String(raw);
    out.push(`<div class="hexmap-tt-extra">${esc(col)}: ${esc(disp)}</div>`);
  }
  return out.join("");
}

function buildOption(prep, cfg, stopsRgb, rect, zoom, theme) {
  const stroke = themeStroke(theme);
  const muted = themeMuted(theme);
  const range = prep.vmax - prep.vmin || 1;

  // Each item carries its centroid coord + value; the full ring is looked up
  // from prep.shapes by dataIndex inside renderItem (api.value can't hold it).
  const data = prep.shapes.map((sh) => [sh.c[0], sh.c[1], sh.v]);

  const renderItem = (params, api) => {
    const sh = prep.shapes[params.dataIndex];
    if (!sh) return null;
    const fill = Number.isFinite(sh.v)
      ? rampColor(stopsRgb, (sh.v - prep.vmin) / range)
      : muted;
    if (sh.kind === "dot") {
      const c = api.coord(sh.proj[0]);
      return {
        type: "circle",
        shape: { cx: c[0], cy: c[1], r: 2.5 },
        style: { fill, stroke, lineWidth: 0.4 },
      };
    }
    const points = sh.proj.map((pt) => api.coord(pt));
    return {
      type: "polygon",
      shape: { points },
      style: { fill, stroke, lineWidth: 0.5 },
    };
  };

  return {
    animation: false,
    grid: {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    },
    xAxis: {
      type: "value",
      show: false,
      scale: true,
      min: prep.ext.xmin,
      max: prep.ext.xmax,
    },
    yAxis: {
      type: "value",
      show: false,
      scale: true,
      min: prep.ext.ymin,
      max: prep.ext.ymax,
    },
    // Wheel zooms, drag pans, pinch zooms on touch — one inside zoom per axis,
    // both at the same start/end so the aspect stays locked. filterMode 'none'
    // keeps every hex drawn (never dropped) while panning past the edge.
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        preventDefaultMouseMove: true,
        start: zoom.sx,
        end: zoom.ex,
      },
      {
        type: "inside",
        yAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        preventDefaultMouseMove: true,
        start: zoom.sy,
        end: zoom.ey,
      },
    ],
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (p) => tooltipHtml(prep.shapes[p.dataIndex], cfg),
    },
    series: [
      {
        type: "custom",
        // Single-pass for the ~2.5k hexes here; chunks only for much larger
        // grids (keeps the current data flicker-free while staying scalable).
        progressive: 2000,
        progressiveThreshold: 4000,
        // Hover is needed for the tooltip, but skip the emphasis restyle of
        // every element (cheap hit-testing, no per-hover repaint of 2.5k items).
        emphasis: { disabled: true },
        data,
        renderItem,
      },
    ],
  };
}

/* ------------------------------- legend footer --------------------------- */

function fmtLegend(v) {
  if (!Number.isFinite(v)) return "";
  const abs = Math.abs(v);
  return abs >= 100 ? v.toFixed(0) : v.toFixed(2);
}

function updateLegend(el, stopsHex, vmin, vmax, unit) {
  if (!el) return;
  const gradient = `linear-gradient(to right, ${stopsHex.join(", ")})`;
  const u = unit || "";
  el.innerHTML =
    `<span class="hexmap-legend-label">${esc(fmtLegend(vmin))}${esc(u)}</span>` +
    `<span class="hexmap-legend-bar" style="background:${gradient}"></span>` +
    `<span class="hexmap-legend-label">${esc(fmtLegend(vmax))}${esc(u)}</span>`;
}

/* --------------------------------- error --------------------------------- */

function showError(el, msg) {
  const region = el.querySelector("[data-hexmap-canvas]");
  if (!region) return;
  const inst = el._hexInst;
  if (inst && !(inst.isDisposed && inst.isDisposed())) {
    try {
      inst.dispose();
    } catch (e) {
      /* ignore */
    }
  }
  el._hexInst = null;
  region.innerHTML = `<div class="hexmap-error">${esc(msg)}</div>`;
}

/* --------------------------------- init ---------------------------------- */

function readZoom(inst) {
  try {
    const dz = inst.getOption().dataZoom || [];
    return {
      sx: dz[0] && dz[0].start != null ? dz[0].start : 0,
      ex: dz[0] && dz[0].end != null ? dz[0].end : 100,
      sy: dz[1] && dz[1].start != null ? dz[1].start : 0,
      ey: dz[1] && dz[1].end != null ? dz[1].end : 100,
    };
  } catch (e) {
    return { sx: 0, ex: 100, sy: 0, ey: 100 };
  }
}

// Wait until the Alpine `filters` store exists — the built-in stores are set up
// on `alpine:init` (store.js), and reading the store inside Alpine.effect below
// is what registers the reactive dependency that drives filter re-fetch. Poll
// (rather than a lone `alpine:init` listener) so ordering vs store setup can't
// race; fall back after ~10s so the map still renders base data if Alpine stalls.
function waitForStore(cb) {
  if (window.Alpine && Alpine.store && Alpine.store("filters")) {
    cb();
    return;
  }
  let tries = 0;
  const iv = setInterval(() => {
    if (window.Alpine && Alpine.store && Alpine.store("filters")) {
      clearInterval(iv);
      cb();
    } else if (++tries > 200) {
      clearInterval(iv);
      cb();
    }
  }, 50);
}

function initOne(el) {
  if (el._hexInit) return;
  el._hexInit = true;

  let cfg;
  try {
    cfg = JSON.parse(el.dataset.config || "{}");
  } catch (e) {
    showError(el, "HexMap: invalid configuration");
    return;
  }

  const region = el.querySelector("[data-hexmap-canvas]");
  const legendEl = el.querySelector("[data-hexmap-legend]");
  if (!region) return;

  const stopsHex = SCHEMES[cfg.scheme] || SCHEMES.heat;
  const stopsRgb = stopsHex.map(hexToRgb);

  let prep = null;

  function ensureInst() {
    const cur = el._hexInst;
    if (cur && !(cur.isDisposed && cur.isDisposed())) return cur;
    // Clear the skeleton (or a prior error) before ECharts takes the container.
    region
      .querySelectorAll(".dashdown-chart-skeleton, .hexmap-error")
      .forEach((n) => n.remove());
    const inst = echarts.init(region, currentEChartsTheme());
    el._hexInst = inst;
    inst.on("dataZoom", () => {
      el._hexZoom = readZoom(inst);
    });
    return inst;
  }

  function render(preserveZoom) {
    if (!prep) return;
    const inst = ensureInst();
    const w = region.clientWidth || 600;
    const h = region.clientHeight || cfg.height || 520;
    const rect = gridRect(prep, w, h);
    const zoom =
      preserveZoom && el._hexZoom
        ? el._hexZoom
        : { sx: 0, ex: 100, sy: 0, ey: 100 };
    inst.setOption(
      buildOption(prep, cfg, stopsRgb, rect, zoom, currentEChartsTheme()),
      true
    );
    updateLegend(legendEl, stopsHex, prep.vmin, prep.vmax, cfg.unit);
    // Introspection hook (verification / debugging): the drawn data's raw lon
    // extent, value range and hex count — never read by the component itself.
    el._hexMeta = {
      lonMin: prep.lonMin,
      lonMax: prep.lonMax,
      vmin: prep.vmin,
      vmax: prep.vmax,
      count: prep.count,
    };
  }

  async function load(filters) {
    try {
      const data = await fetchQueryData(cfg.query, {}, filters);
      if (data && data.error) throw new Error(data.error);
      prep = prepare(recordsOf(data), cfg);
      el._hexZoom = null; // new data (e.g. a city switch) → reset to full extent
      render(false);
    } catch (err) {
      showError(el, `HexMap could not load: ${(err && err.message) || err}`);
    }
  }

  waitForStore(() => {
    // Single reactive path — same pattern chart.js uses. The effect runs once
    // (initial render) and re-runs whenever any filter value changes; spreading
    // the store registers a dependency on every key, so the City dropdown drives
    // a re-fetch + redraw for the new city.
    if (window.Alpine && Alpine.effect) {
      Alpine.effect(() => {
        const filters = { ...(Alpine.store("filters") || {}) };
        load(filters);
      });
    } else {
      load({});
    }

    // Re-theme on the same signal chart.js listens to: dispose + re-init with
    // the new theme, then redraw the (already-fetched) data, preserving zoom.
    onThemeChange(() => {
      const old = el._hexInst;
      if (old && !(old.isDisposed && old.isDisposed())) {
        try {
          old.dispose();
        } catch (e) {
          /* ignore */
        }
      }
      el._hexInst = null;
      render(true);
    });

    // Keep true shape across container resizes (recompute the grid rect) while
    // preserving zoom (dataZoom % is relative to the fixed data extent).
    let rTimer = null;
    const ro = new ResizeObserver(() => {
      clearTimeout(rTimer);
      rTimer = setTimeout(() => {
        const inst = el._hexInst;
        if (!inst || (inst.isDisposed && inst.isDisposed()) || !prep) return;
        const w = region.clientWidth || 600;
        const h = region.clientHeight || cfg.height || 520;
        inst.setOption({ grid: gridRect(prep, w, h) });
        inst.resize();
      }, 80);
    });
    ro.observe(region);
  });
}

function initAll() {
  document
    .querySelectorAll('[data-async-component="hexmap"]')
    .forEach((el) => initOne(el));
}

// ES modules are deferred, so the DOM is parsed when this runs.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initAll);
} else {
  initAll();
}
