# %% [markdown]
# # 02 - Climate, terrain, soil and land cover (Google Earth Engine)
# **REAL mode only.** Downloads everything except Sentinel-2 and InSAR onto the same grid:
#
# | Output file | Source (Earth Engine id) | Notes |
# |---|---|---|
# | `climate/rain_mm_<yr>.tif` | `UCSB-CHG/CHIRPS/DAILY` | monthly total, mm |
# | `climate/temp_c_<yr>.tif`, `pet_mm_<yr>.tif`, `et_mm_<yr>.tif` | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | 2 m temperature (deg C), potential and actual evaporation (mm/month, positive) |
# | `static/elevation_m.tif`, `slope_deg.tif` | `USGS/SRTMGL1_003` | slope computed at 30 m then averaged |
# | `static/clay_pct.tif` | `OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02` | topsoil clay % |
# | `static/upa_km2.tif` | `MERIT/Hydro/v1_0_1` | upstream drainage area (river / wetness proxy) |
# | `static/lake_occurrence.tif` | `JRC/GSW1_4/GlobalSurfaceWater` | % of time under water; >= 50 = lake |
# | `static/landcover_fractions.tif` | `ESA/WorldCover/v200` (2021) | fraction of each class in the cell |
#
# CHIRPS (5 km) and ERA5-Land (11 km) are much coarser than the 100 m grid, so they are interpolated bilinearly; treat them as regional
# climate forcing, not as local rainfall. Files that already exist are skipped.

# %%
# Cell 1 - Environment (identical in every notebook). Runs on Colab, Jupyter, VS Code, PyCharm, Spyder.
import os, sys, json, subprocess, importlib
from pathlib import Path

USE_DRIVE = True      # Colab only: keep project files on Google Drive so they survive a restart


def ensure(*pip_names):
    """Import each package; pip-install it first if it is missing."""
    import_names = {"scikit-learn": "sklearn", "earthengine-api": "ee", "hyp3-sdk": "hyp3_sdk",
                    "asf-search": "asf_search", "pillow": "PIL"}
    for pkg in pip_names:
        try:
            module_name = import_names[pkg] if pkg in import_names else pkg
            importlib.import_module(module_name)
        except ImportError:
            print("Installing", pkg, "...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and USE_DRIVE:
    colab = importlib.import_module("google.colab")
    colab.drive.mount("/content/drive")
    default_dir = "/content/drive/MyDrive/naivasha_project"
elif IN_COLAB:
    default_dir = "/content/naivasha_project"
else:
    cwd = Path.cwd()
    default_dir = cwd.parent if cwd.name in ("notebooks", "scripts") else cwd
PROJECT_DIR = Path(os.environ.get("NAIVASHA_DIR", default_dir))   # override with env var NAIVASHA_DIR
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
print("Running on:", "Google Colab" if IN_COLAB else "local Python", "| project folder:", PROJECT_DIR)
# %%
# Cell 2 - Load the project configuration written by notebook 00
CFG_FILE = PROJECT_DIR / "config.json"
if not CFG_FILE.exists():
    raise FileNotFoundError(f"{CFG_FILE} not found - run 00_setup_and_config first.")
CFG = json.loads(CFG_FILE.read_text())
if not isinstance(CFG.get("aoi"), dict) or not isinstance(CFG["aoi"].get("geometry"), dict):
    raise ValueError(
        f"{CFG_FILE} does not contain the study-area geometry; run 00_setup_and_config "
        "in this project folder first."
    )
GRID = CFG["grid"]
DATA = PROJECT_DIR / CFG["paths"]["data"]
OUT = PROJECT_DIR / CFG["paths"]["outputs"]
MODE = CFG["mode"]                                   # "real" or "demo"
YEARS = list(range(CFG["start_year"], CFG["end_year"] + 1))
MONTHS = [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]
print(f"Mode: {MODE} | years {YEARS[0]}-{YEARS[-1]} | grid {GRID['width']} x {GRID['height']} px @ {GRID['res']} m")
if MODE == "demo":
    print(">>> DEMO MODE: all data are SYNTHETIC. Outputs are for testing the pipeline only, not scientific results.")
# %%
# Cell - Connect to Google Earth Engine (a browser sign-in link appears the first time)
ensure("earthengine-api", "requests")
import ee
GEE_PROJECT = CFG["gee_project"]
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()                 # follow the link, paste the code
    ee.Initialize(project=GEE_PROJECT)
print("Earth Engine ready, project:", GEE_PROJECT)
# %%
# Cell - Download helpers: pull an Earth Engine image onto OUR grid in tiles and save one GeoTIFF
import io, time, requests
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

NODATA = -9999.0
AOI_GRID_GEOMETRY = transform_geom("EPSG:4326", GRID["crs"], CFG["aoi"]["geometry"])


def tile_windows(width, height, tile=512):
    """Pixel windows (col0, row0, w, h) that cover the whole grid."""
    return [(c, r, min(tile, width - c), min(tile, height - r))
            for r in range(0, height, tile) for c in range(0, width, tile)]


def fetch_tile(image, g, col0, row0, w, h, retries=4):
    """Download one tile of an ee.Image as a numpy array [bands, rows, cols]."""
    res = g["res"]
    x0, y1 = g["xmin"] + col0 * res, g["ymax"] - row0 * res
    region = ee.Geometry.Rectangle([x0, y1 - h * res, x0 + w * res, y1], proj=g["crs"], geodesic=False)
    params = {"region": region, "crs": g["crs"], "crs_transform": [res, 0, x0, 0, -res, y1], "format": "GEO_TIFF"}
    for attempt in range(retries):
        try:
            r = requests.get(image.getDownloadURL(params), timeout=900)
            r.raise_for_status()
            with rasterio.MemoryFile(r.content) as mem, mem.open() as src:
                return src.read().astype("float32")
        except Exception as err:
            print(f"   tile retry {attempt + 1}/{retries}: {str(err)[:120]}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Tile download failed - try a smaller tile size or re-run (finished files are skipped).")


def download_image(image, out_path, names, g, tile=512):
    """Download `image` (bands must match `names`) onto grid `g`; skip if the file already exists."""
    out_path = Path(out_path)
    if out_path.exists():
        print("   exists, skipped:", out_path.name)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full = np.full((len(names), g["height"], g["width"]), NODATA, dtype="float32")
    for (c0, r0, w, h) in tile_windows(g["width"], g["height"], tile):
        arr = fetch_tile(image, g, c0, r0, w, h)
        if arr.shape[0] != len(names):
            raise ValueError(f"Expected {len(names)} bands, got {arr.shape[0]}")
        hh, ww = min(h, arr.shape[1]), min(w, arr.shape[2])      # guard against 1-pixel rounding at tile edges
        full[:, r0:r0 + hh, c0:c0 + ww] = arr[:, :hh, :ww]
    transform = from_origin(g["xmin"], g["ymax"], g["res"], g["res"])
    outside_aoi = geometry_mask(
        [AOI_GRID_GEOMETRY], out_shape=(g["height"], g["width"]), transform=transform
    )
    full[:, outside_aoi] = NODATA
    with rasterio.open(out_path, "w", driver="GTiff", height=g["height"], width=g["width"], count=len(names),
                       dtype="float32", crs=g["crs"], transform=transform, nodata=NODATA, compress="deflate") as dst:
        dst.write(full)
        dst.descriptions = tuple(names)
    print("   saved:", out_path.name)
# %%
# Cell 3 - Raster helpers (read / write multi-band GeoTIFFs that sit on the project grid)
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

NODATA = -9999.0
TRANSFORM = from_origin(GRID["xmin"], GRID["ymax"], GRID["res"], GRID["res"])
H, W = GRID["height"], GRID["width"]
AOI_GRID_GEOMETRY = transform_geom("EPSG:4326", GRID["crs"], CFG["aoi"]["geometry"])


def read_stack(path):
    """Read an analysis-grid GeoTIFF and mask every pixel outside the setup AOI."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is missing - run the earlier notebook that creates it.")
    with rasterio.open(path) as src:
        expected_crs = rasterio.crs.CRS.from_string(GRID["crs"])
        if src.crs != expected_crs or src.transform != TRANSFORM or (src.height, src.width) != (H, W):
            raise ValueError(
                f"{path} is not aligned to the configured analysis grid "
                f"({GRID['crs']}, {W} x {H}, {TRANSFORM})."
            )
        arr = src.read().astype("float32")
        names = list(src.descriptions)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    arr[arr == NODATA] = np.nan
    outside_aoi = geometry_mask([AOI_GRID_GEOMETRY], out_shape=(H, W), transform=TRANSFORM)
    arr[:, outside_aoi] = np.nan
    return arr, names


def write_stack(path, arr, names=None, dtype="float32"):
    """Write [rows, cols] or [bands, rows, cols] to a GeoTIFF on the project grid (NaN -> nodata)."""
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[-2:] != (H, W):
        raise ValueError(f"Expected raster data shaped [bands, {H}, {W}], got {arr.shape}.")
    outside_aoi = geometry_mask([AOI_GRID_GEOMETRY], out_shape=(H, W), transform=TRANSFORM)
    arr[:, outside_aoi] = np.nan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(arr), arr, NODATA).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", height=out.shape[1], width=out.shape[2],
                       count=out.shape[0], dtype=dtype, crs=GRID["crs"], transform=TRANSFORM,
                       nodata=NODATA, compress="deflate") as dst:
        dst.write(out)
        if names:
            dst.descriptions = tuple(names)


def read_years(folder, name):
    """Read name_YYYY.tif for every year and stack them in time -> [months, rows, cols]."""
    return np.concatenate([read_stack(Path(folder) / f"{name}_{y}.tif")[0] for y in YEARS], axis=0)
# %%
# Cell 6 - Settings
AOI = ee.Geometry(CFG["aoi"]["geometry"])
CRS = GRID["crs"]
TILE = 512
month_names = lambda y: [f"{y}-{m:02d}" for m in range(1, 13)]
print("ready; years", YEARS[0], "-", YEARS[-1])

# %% [markdown]
# ## 1. Rainfall (CHIRPS monthly totals)

# %%
# Cell 7 - One 12-band image per year: monthly rainfall in mm
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")


def rain_year(year):
    imgs = []
    for m in range(1, 13):
        s = ee.Date.fromYMD(year, m, 1)
        imgs.append(chirps.filterDate(s, s.advance(1, "month")).sum().rename(f"{year}-{m:02d}"))
    return ee.Image.cat(imgs).resample("bilinear").toFloat().unmask(NODATA)


for y in YEARS:
    download_image(rain_year(y), DATA / "climate" / f"rain_mm_{y}.tif", month_names(y), GRID, tile=TILE)

# %% [markdown]
# ## 2. ERA5-Land: temperature, potential evaporation, evaporation

# %%
# Cell 8 - Monthly means/sums; ERA5-Land reports evaporation as negative metres, so we take abs() and convert to mm
era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
ERA_VARS = {"temp_c": ("temperature_2m", lambda i: i.subtract(273.15)),
            "pet_mm": ("potential_evaporation_sum", lambda i: i.abs().multiply(1000)),
            "et_mm": ("total_evaporation_sum", lambda i: i.abs().multiply(1000))}


def era_year(year, band, fn):
    col = era.filterDate(f"{year}-01-01", f"{year + 1}-01-01").select(band).sort("system:time_start")
    return fn(col.toBands()).rename(month_names(year)).resample("bilinear").toFloat().unmask(NODATA)


for name, (band, fn) in ERA_VARS.items():
    print("==", name)
    for y in YEARS:
        download_image(era_year(y, band, fn), DATA / "climate" / f"{name}_{y}.tif", month_names(y), GRID, tile=TILE)

# %% [markdown]
# ## 3. Terrain, soil, hydrology and lake

# %%
# Cell 9 - Elevation and slope (SRTM 30 m, averaged to the grid)
srtm = ee.Image("USGS/SRTMGL1_003")
slope30 = ee.Terrain.slope(srtm)
avg = lambda img: img.reduceResolution(ee.Reducer.mean(), maxPixels=1024)
download_image(avg(srtm).rename("elevation_m").toFloat().unmask(NODATA), DATA / "static" / "elevation_m.tif", ["elevation_m"], GRID, TILE)
download_image(avg(slope30).rename("slope_deg").toFloat().unmask(NODATA), DATA / "static" / "slope_deg.tif", ["slope_deg"], GRID, TILE)

# %%
# Cell 10 - Clay content, upstream area and lake occurrence
clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("clay_pct")
upa = ee.Image("MERIT/Hydro/v1_0_1").select("upa").rename("upa_km2")
water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).rename("lake_occurrence")
download_image(clay.resample("bilinear").toFloat().unmask(NODATA), DATA / "static" / "clay_pct.tif", ["clay_pct"], GRID, TILE)
download_image(upa.resample("bilinear").toFloat().unmask(NODATA), DATA / "static" / "upa_km2.tif", ["upa_km2"], GRID, TILE)
download_image(avg(water).toFloat().unmask(NODATA), DATA / "static" / "lake_occurrence.tif", ["lake_occurrence"], GRID, TILE)

# %% [markdown]
# ## 4. Land cover fractions (ESA WorldCover 2021)
# Class codes: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 80 water. Each output band is the fraction (0-1) of 10 m pixels of that
# class inside each grid cell, which is more informative than a single "majority" class on a 100 m cell.

# %%
# Cell 11 - Fractions
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
CLASSES = {"crop": 40, "grass": 30, "tree": 10, "shrub": 20, "built": 50, "bare": 60, "water": 80}
frac = ee.Image.cat([wc.eq(code).rename(name) for name, code in CLASSES.items()]).toFloat()
frac = frac.reduceResolution(ee.Reducer.mean(), maxPixels=1024).unmask(NODATA)
download_image(frac, DATA / "static" / "landcover_fractions.tif", list(CLASSES), GRID, TILE)

# %% [markdown]
# ## 5. Quality check

# %%
# Cell 12 - Plot every static layer and a rainfall time series
import matplotlib.pyplot as plt
if "read_stack" not in globals():
    import numpy as np
    import rasterio

    def read_stack(path):
        with rasterio.open(path) as src:
            arr = src.read().astype("float32")
            names = list(src.descriptions)
        arr[arr == -9999.0] = np.nan
        return arr, names

fig, ax = plt.subplots(2, 4, figsize=(18, 8))
layers = [("elevation_m", "terrain"), ("slope_deg", "magma"), ("clay_pct", "YlOrBr"), ("upa_km2", "Blues"), ("lake_occurrence", "Blues")]
for a, (name, cmap) in zip(ax.ravel(), layers):
    arr = read_stack(DATA / "static" / f"{name}.tif")[0][0]
    im = a.imshow(np.log10(arr + 1) if name == "upa_km2" else arr, cmap=cmap); a.set_title(name); a.axis("off"); plt.colorbar(im, ax=a, shrink=0.7)
lc, lcn = read_stack(DATA / "static" / "landcover_fractions.tif")
ax.ravel()[5].imshow(lc[lcn.index("crop")], cmap="Greens"); ax.ravel()[5].set_title("cropland fraction"); ax.ravel()[5].axis("off")
rain = np.concatenate([np.nanmean(read_stack(DATA / "climate" / f"rain_mm_{y}.tif")[0], axis=(1, 2)) for y in YEARS])
ax.ravel()[6].bar(range(len(rain)), rain); ax.ravel()[6].set_title("Mean monthly rain (mm)")
ax.ravel()[7].axis("off")
fig.savefig(OUT / "figures" / "02_static_layers.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Sanity checks to eyeball:** the lake must show as high `lake_occurrence`; elevation should be about 1,880-2,000 m near the shore and rise
# toward Longonot, the Mau escarpment and the Aberdares; cropland fraction should hug the lake margin. If the picture is blank or flipped,
# the bbox or the project id is wrong.
#
# **Optional inputs** (put in `data/ancillary/`; notebook 05 uses them automatically): `boreholes.csv` (lon, lat, optional abstraction),
# `gnss.csv` (lon, lat, vel_up_mm_yr), `geology.gpkg` (polygons with a text column `unit`).
#
# **Next:** `03_insar_hyp3_download`.
