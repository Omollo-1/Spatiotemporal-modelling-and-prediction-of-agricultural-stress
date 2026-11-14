# %% [markdown]
# # 01 - Sentinel-2 vegetation and moisture indices (Google Earth Engine)
# **REAL mode only.** Builds monthly cloud-free composites of Sentinel-2 and downloads five indices as GeoTIFFs on the project grid.
#
# | Index | Formula (Sentinel-2 bands) | What it tells you |
# |---|---|---|
# | NDVI | (B8 - B4) / (B8 + B4) | greenness / biomass |
# | NDMI | (B8 - B11) / (B8 + B11) | canopy water content |
# | MSI  | B11 / B8 | moisture stress (higher = drier) |
# | NDRE | (B8A - B5) / (B8A + B5) | chlorophyll / crop vigour |
# | EVI  | 2.5 (B8 - B4) / (B8 + 6 B4 - 7.5 B2 + 1) | greenness, less saturation |
#
# **Design choices (see docs/01_methodology.md):**
# * `COPERNICUS/S2_HARMONIZED` (Level-1C top-of-atmosphere) is used for the whole period, because Level-2A coverage over Africa is incomplete
#   before about 2018. One consistent product across 2015-2024 matters more than atmospheric correction, because stress is measured
#   as an *anomaly against the same product's own climatology*.
# * Clouds are masked with **Cloud Score+** (`cs_cdf >= 0.6`).
# * Each month = per-pixel **median** of clear observations, averaged (not sampled) from 10-20 m to the project grid.
# * Output: `data/s2/<INDEX>_<YEAR>.tif`, 12 bands (one per month), nodata -9999. Already-downloaded files are skipped, so you can stop and resume.
#
# Expect roughly 30-90 minutes for 10 years x 5 indices at 100 m.

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
# Cell 6 - Settings
INDICES = ["NDVI", "NDMI", "MSI", "NDRE", "EVI"]
CLOUD_THRESHOLD = 0.6                        # Cloud Score+ cs_cdf: higher = clearer. 0.6 is a common balanced choice.
TILE = 512                                   # pixels per download tile (reduce to 256 if you see size errors)
AOI = ee.Geometry(CFG["aoi"]["geometry"])
CRS = GRID["crs"]
print("AOI:", CFG["aoi"]["name"], "| bounds:", CFG["bbox"], "| indices:", INDICES)

# %% [markdown]
# ## 1. Index functions

# %%
# Cell 7 - Add the five indices to a Sentinel-2 image
def add_indices(img):
    r = img.select(["B2", "B4", "B5", "B8", "B8A", "B11"]).divide(10000)
    b2, b4, b5, b8, b8a, b11 = [r.select(b) for b in ["B2", "B4", "B5", "B8", "B8A", "B11"]]
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
    ndmi = b8.subtract(b11).divide(b8.add(b11)).rename("NDMI")
    msi = b11.divide(b8).rename("MSI")
    ndre = b8a.subtract(b5).divide(b8a.add(b5)).rename("NDRE")
    evi = b8.subtract(b4).multiply(2.5).divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)).rename("EVI")
    return img.addBands([ndvi, ndmi, msi, ndre, evi])

print("index function defined")

# %% [markdown]
# ## 2. Monthly composites

# %%
# Cell 8 - Collections: Sentinel-2 L1C + Cloud Score+ (clear-sky score per image)
S2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(AOI)
CSP = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")


def monthly_composite(year, month, index):
    """Median of clear observations in one month, area-averaged to the project grid. Empty month -> fully masked image."""
    start = ee.Date.fromYMD(year, month, 1)
    col = (S2.filterDate(start, start.advance(1, "month")).linkCollection(CSP, ["cs_cdf"])
           .map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLOUD_THRESHOLD)))
           .map(add_indices).select(index))
    empty = ee.Image.constant(0).rename(index).updateMask(ee.Image.constant(0))
    comp = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), empty))
    # aggregate 10-20 m pixels into the output cell by mean instead of picking one sample
    return (comp.setDefaultProjection(ee.Projection(CRS).atScale(10))
            .reduceResolution(ee.Reducer.mean(), bestEffort=False, maxPixels=1024)
            .rename(f"{year}-{month:02d}"))


def year_image(year, index):
    return ee.Image.cat([monthly_composite(year, m, index) for m in range(1, 13)]).toFloat().unmask(NODATA)

print("composite functions defined")

# %% [markdown]
# ## 3. Quick test on one small tile (do this before the long run)

# %%
# Cell 9 - Download a 64 x 64 px tile of NDVI for July 2020 and look at it
test_img = year_image(2020, "NDVI")
arr = fetch_tile(test_img, GRID, 0, 0, min(64, GRID["width"]), min(64, GRID["height"]))
july = np.where(arr[6] == NODATA, np.nan, arr[6])
print("tile shape", arr.shape, "| July NDVI valid share %.2f | range %.2f to %.2f" % (np.isfinite(july).mean(), np.nanmin(july), np.nanmax(july)))
assert np.isfinite(july).any(), "No valid pixels - check the project id, catchment geometry and dates"

# %% [markdown]
# ## 4. Download everything (resumable)

# %%
# Cell 10 - Loop over indices and years. Safe to stop and re-run: finished files are skipped.
names = lambda y: [f"{y}-{m:02d}" for m in range(1, 13)]
for index in INDICES:
    print("==", index)
    for year in YEARS:
        download_image(year_image(year, index), DATA / "s2" / f"{index}_{year}.tif", names(year), GRID, tile=TILE)

# %% [markdown]
# ## 5. Quality check

# %%
# Cell 11 - How many months per year have usable data? (Sentinel-2 starts mid-2015; cloudy months stay empty)
rows = []
for y in YEARS:
    a, _ = read_stack(DATA / "s2" / f"NDVI_{y}.tif")
    rows.append({"year": y, **{f"m{m + 1:02d}": round(float(np.isfinite(a[m]).mean()), 2) for m in range(12)}})
qa = pd.DataFrame(rows).set_index("year")
qa

# %%
# Cell 12 - Visual check: valid-pixel share by month, and a mean-NDVI map
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
im = ax[0].imshow(qa.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto"); ax[0].set_xticks(range(12)); ax[0].set_xticklabels(range(1, 13))
ax[0].set_yticks(range(len(YEARS))); ax[0].set_yticklabels(YEARS); ax[0].set_title("Share of valid NDVI pixels"); plt.colorbar(im, ax=ax[0])
mean_ndvi = np.nanmean(np.concatenate([read_stack(DATA / "s2" / f"NDVI_{y}.tif")[0] for y in YEARS]), axis=0)
im = ax[1].imshow(mean_ndvi, cmap="YlGn", vmin=0, vmax=0.8); ax[1].set_title("Mean NDVI 2015-2024"); ax[1].axis("off"); plt.colorbar(im, ax=ax[1])
fig.savefig(OUT / "figures" / "01_s2_quality.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Next:** `02_climate_terrain_landcover`. Months that are mostly empty (heavy cloud) are gap-filled up to 2 months in notebook 05;
# longer gaps stay empty and the model handles them as missing values.
