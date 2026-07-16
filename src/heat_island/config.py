"""Pipeline configuration. A single PipelineConfig instance is threaded through all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Canonical model feature list — order matters (model training, SHAP, dashboard all use it).
FEATURES: list[str] = [
    "ndvi",
    "ndbi",
    "ndwi",
    "albedo",
    "elevation",
    "building_density",
    "road_density",
    "dist_water_m",
    "dist_park_m",
]

# Demographic columns are nullable everywhere (US-only enrichment).
DEMOGRAPHIC_COLS: list[str] = ["median_income", "pct_over_65", "pct_under_5"]

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ESA WorldCover class -> "plantable" weight: how much of a pixel of this class could
# realistically take new tree canopy. Grass/shrub/crop/bare take trees outright; built-up
# gets a small street-pit/depaving credit; existing tree cover (10) offers no ADDITIONAL
# room (its cooling is already in the observed LST); water/ice/wetland/moss take none.
LANDCOVER_TREE_CLASS = 10
PLANTABLE_CLASS_WEIGHTS: dict[int, float] = {
    10: 0.0,   # tree cover (already canopy)
    20: 1.0,   # shrubland
    30: 1.0,   # grassland
    40: 1.0,   # cropland
    50: 0.15,  # built-up (tree pits / depaving potential only)
    60: 1.0,   # bare / sparse vegetation
    70: 0.0,   # snow & ice
    80: 0.0,   # permanent water
    90: 0.0,   # herbaceous wetland
    95: 0.0,   # mangroves
    100: 0.0,  # moss & lichen
}


@dataclass
class PipelineConfig:
    # spatial unit
    h3_resolution: int = 9

    # satellite search
    years_back: int = 3            # number of completed summers to composite
    max_cloud_pct: int = 20        # STAC eo:cloud_cover filter
    relaxed_cloud_pct: int = 40    # one-shot fallback when nothing found
    landsat_max_scenes: int = 18
    s2_max_scenes: int = 24
    raster_res_deg: float = 0.00027   # ~30 m at the equator (Landsat / DEM load grid)
    s2_res_deg: float = 0.00018       # ~20 m (Sentinel-2 load grid)
    landcover_res_deg: float = 0.0001  # ~11 m (ESA WorldCover native is 1/12000 deg)
    bbox_pad_deg: float = 0.005

    # OSM
    buffer_m: float = 2000.0          # fetch water/parks slightly beyond the boundary
    dist_cap_m: float = 10000.0       # distance features capped here (and NaN-filled)
    min_water_area_m2: float = 10000.0  # "large" water bodies only (≥ 1 ha)

    # modelling
    cv_parent_offset: int = 2      # spatial CV blocks = H3 parent at (res - offset)
    cv_folds: int = 5
    min_hexes: int = 30
    min_park_hexes: int = 20       # fewer than this → NDVI-percentile fallback for target
    park_ndvi_quantile: float = 0.75
    fallback_ndvi_quantile: float = 0.90
    seed: int = 42

    # paths
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "heat.duckdb"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def city_cache(self, city_id: str) -> Path:
        p = self.cache_dir / city_id
        p.mkdir(parents=True, exist_ok=True)
        return p
