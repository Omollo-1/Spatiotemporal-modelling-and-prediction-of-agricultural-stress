# %% [markdown]
# # 04 - InSAR time series with MintPy -> vertical velocity and displacement
# **REAL mode only.** Takes the HyP3 interferograms from notebook 03 and runs the MintPy small-baseline (SBAS) workflow for each
# track: network check, phase-unwrapping error correction, time-series inversion, tropospheric and DEM-error corrections, velocity.
# Then it converts line-of-sight (LOS) motion to **vertical** motion and writes the three files that notebook 05 reads:
#
# * `data/insar/velocity_vertical_mm_yr.tif` (negative = subsiding)
# * `data/insar/velocity_std_mm_yr.tif`
# * `data/insar/disp_vertical_monthly_mm.tif` (monthly displacement, mm, relative to the first epoch)
#
# **Key assumption (state it in your methods):** one LOS direction cannot separate vertical from horizontal motion. Here vertical =
# LOS / cos(incidence angle), i.e. horizontal motion is assumed small. This is reasonable for subsidence bowls caused by fluid withdrawal, but
# *not* for tectonic or volcanic motion in the rift. With both ascending and descending tracks the notebook averages the two vertical
# estimates and reports their difference as a consistency check; for a full east-west/vertical decomposition use MintPy's `asc_desc2horz_vert.py`.
#
# **Other things to know:** MintPy installs with `pip install mintpy` on most Linux/Colab setups; on Windows use WSL or conda
# (`conda install -c conda-forge mintpy`). Processing is memory-hungry: crop to the study area (done here) and expect tens of minutes per track.

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
# Cell 4 - Packages and settings
if os.name == "nt":
    try:
        import mintpy
    except ImportError as err:
        raise RuntimeError(
            "MintPy is not installed in this Python environment. On Windows, create or update the "
            "'naivasha' Conda environment from environment.yml, then select that environment as the notebook kernel."
        ) from err
ensure("mintpy", "h5py", "rasterio", "pyproj", "scipy")
import glob, shutil, zipfile, subprocess
import h5py
from affine import Affine
from shapely.geometry import shape
from rasterio.windows import Window, from_bounds
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling, transform_geom
from scipy.interpolate import interp1d

INSAR = DATA / "insar"
TRACKS = {"asc": INSAR / "hyp3_asc", "desc": INSAR / "hyp3_desc"}
TRACKS = {k: v for k, v in TRACKS.items() if v.exists() and any(v.glob("*.zip"))}
REF_LALO = None                      # e.g. "-0.62,36.55" = a stable, high-coherence point (outside farms, lake, volcanoes); None = MintPy picks one
print("tracks with data:", list(TRACKS) or "NONE - run notebook 03 first")

# %% [markdown]
# ## 1. Unzip and crop every product to the same area
# MintPy needs all interferograms to have identical size. HyP3 products from one frame share a pixel grid, so we cut each to the
# overlap of all products and the study area.

# %%
# Cell 5 - Unzip
for tag, folder in TRACKS.items():
    for z in sorted(folder.glob("*.zip")):
        target = folder / z.stem
        if not target.exists():
            zipfile.ZipFile(z).extractall(folder)
    print(tag, "->", len(list(folder.glob("*/"))), "product folders")

# %%
# Cell 6 - Crop helper: common bounds of all products intersected with the AOI (in the products' own UTM zone)
def crop_products(folder, suffixes=("unw_phase", "corr", "dem", "lv_theta", "lv_phi"), aoi=CFG["aoi"]["geometry"]):
    probe = sorted(glob.glob(str(folder / "*" / "*_unw_phase.tif")))
    if not probe:
        raise FileNotFoundError(f"No *_unw_phase.tif under {folder}")
    with rasterio.open(probe[0]) as src:
        crs = src.crs
    aoi_product = transform_geom("EPSG:4326", crs, aoi)
    aoi_bounds = shape(aoi_product).bounds
    bounds = [rasterio.open(f).bounds for f in probe]
    left, bottom = max(max(b.left for b in bounds), aoi_bounds[0]), max(max(b.bottom for b in bounds), aoi_bounds[1])
    right, top = min(min(b.right for b in bounds), aoi_bounds[2]), min(min(b.top for b in bounds), aoi_bounds[3])
    if left >= right or bottom >= top:
        raise ValueError("The AOI does not overlap the products - check the track/frame chosen in notebook 03.")
    shapes = set()
    for suf in suffixes:
        for f in glob.glob(str(folder / "*" / f"*_{suf}.tif")):
            out = f.replace(f"_{suf}.tif", f"_{suf}_clip.tif")
            if Path(out).exists():
                continue
            with rasterio.open(f) as src:
                w = from_bounds(left, bottom, right, top, transform=src.transform)
                win = Window(int(round(w.col_off)), int(round(w.row_off)), int(round(w.width)), int(round(w.height)))
                data = src.read(window=win)
                crop_transform = src.window_transform(win)
                outside_aoi = geometry_mask(
                    [aoi_product], out_shape=data.shape[-2:], transform=crop_transform
                )
                nodata = src.nodata
                if nodata is None:
                    if not np.issubdtype(data.dtype, np.floating):
                        raise ValueError(f"{f} has no nodata value and is not floating-point; cannot mask outside the catchment.")
                    nodata = np.nan
                data[:, outside_aoi] = nodata
                prof = src.profile | dict(width=win.width, height=win.height, transform=crop_transform, nodata=nodata)
            with rasterio.open(out, "w", **prof) as dst:
                dst.write(data)
            shapes.add(data.shape[1:])
    if len(shapes) > 1:
        raise ValueError(f"Cropped products differ in size {shapes}: the interferograms are not on a common grid.")
    return (left, bottom, right, top), shapes


for tag, folder in TRACKS.items():
    print(tag, crop_products(folder))

# %% [markdown]
# ## 2. MintPy time-series analysis

# %%
# Cell 7 - Write one MintPy config per track
CONFIG_TEMPLATE = """
mintpy.load.processor      = hyp3
mintpy.load.unwFile        = {d}/*/*_unw_phase_clip.tif
mintpy.load.corFile        = {d}/*/*_corr_clip.tif
mintpy.load.demFile        = {d}/*/*_dem_clip.tif
mintpy.load.incAngleFile   = {d}/*/*_lv_theta_clip.tif
mintpy.load.azAngleFile    = {d}/*/*_lv_phi_clip.tif
mintpy.reference.lalo      = {ref}
mintpy.unwrapError.method  = bridging
mintpy.topographicResidual = yes
mintpy.troposphericDelay.method = height_correlation
mintpy.deramp              = linear
mintpy.networkInversion.weightFunc = var
"""
workdirs = {}
for tag, folder in TRACKS.items():
    wd = INSAR / f"mintpy_{tag}"
    wd.mkdir(exist_ok=True)
    (wd / "hyp3.cfg").write_text(CONFIG_TEMPLATE.format(d=folder.resolve(), ref=REF_LALO or "auto"))
    workdirs[tag] = wd
    print("wrote", wd / "hyp3.cfg")

# %%
# Cell 8 - Run MintPy (slow: minutes to hours). Output is streamed; re-running resumes where files already exist.
for tag, wd in workdirs.items():
    if (wd / "velocity.h5").exists():
        print(tag, "velocity.h5 exists - skipping run (delete it to re-run)")
        continue
    exe = shutil.which("smallbaselineApp.py") or "smallbaselineApp.py"
    print("== MintPy", tag)
    subprocess.run([exe, "hyp3.cfg"], cwd=wd, check=True)

# %% [markdown]
# **Check MintPy's own diagnostics** in each `mintpy_*/pic/` folder: coherence, unwrapping-error counts, the tropospheric-delay fit and
# the velocity map. If the velocity is dominated by a smooth ramp or by one noisy interferogram, fix the reference point or drop the
# bad pair before trusting anything downstream.

# %% [markdown]
# ## 3. Convert LOS to vertical and put it on the project grid

# %%
# Cell 9 - Helpers: read a MintPy HDF5 layer with its geo-information, reproject onto the project grid
def read_h5(path, dset):
    with h5py.File(path, "r") as f:
        arr = f[dset][:].astype("float32")
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        dates = [d.decode() for d in f["date"][:]] if "date" in f else None
    return arr, attrs, dates


def to_grid(arr, attrs, fallback_crs=GRID["crs"]):
    """Area-average a MintPy (geocoded) array onto the project grid."""
    tf = Affine(float(attrs["X_STEP"]), 0, float(attrs["X_FIRST"]), 0, float(attrs["Y_STEP"]), float(attrs["Y_FIRST"]))
    crs = f"EPSG:{attrs['EPSG']}" if "EPSG" in attrs else fallback_crs
    out = np.full((H, W), np.nan, dtype="float32")
    reproject(arr, out, src_transform=tf, src_crs=crs, dst_transform=TRANSFORM, dst_crs=GRID["crs"],
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return out

print("helpers ready")

# %%
# Cell 10 - Per track: velocity (mm/yr), its std, and monthly displacement (mm), all vertical-assumption and on the grid
def process_track(wd):
    geom = sorted(glob.glob(str(wd / "inputs" / "geometry*.h5")))[0]
    inc_deg, _, _ = read_h5(geom, "incidenceAngle")
    cos_inc = np.cos(np.deg2rad(inc_deg))
    vel, attrs, _ = read_h5(wd / "velocity.h5", "velocity")                       # m/yr, LOS, + = towards satellite
    vel_up = vel * 1000 / cos_inc                                                  # mm/yr, + = uplift
    with h5py.File(wd / "velocity.h5", "r") as f:
        vstd = f["velocityStd"][:].astype("float32") if "velocityStd" in f else np.full_like(vel, np.nan)
    ts, tattrs, dates = read_h5(wd / "timeseries.h5", "timeseries")               # m, LOS, relative to first date
    ts_up = ts * 1000 / cos_inc[None]
    t_obs = np.array([(pd.Timestamp(d) - pd.Timestamp(dates[0])).days for d in dates], dtype=float)
    mid = np.array([(pd.Timestamp(f"{m}-15") - pd.Timestamp(dates[0])).days for m in MONTHS], dtype=float)
    f = interp1d(t_obs, ts_up.reshape(len(t_obs), -1), axis=0, bounds_error=False, fill_value=np.nan)
    ts_month = f(mid).reshape((len(MONTHS),) + vel.shape)                         # NaN outside the observed period
    grid_ts = np.stack([to_grid(ts_month[i], attrs) for i in range(len(MONTHS))])
    return to_grid(vel_up, attrs), to_grid(vstd * 1000 / cos_inc, attrs), grid_ts


results = {tag: process_track(wd) for tag, wd in workdirs.items()}
print("processed tracks:", list(results))

# %%
# Cell 11 - Combine tracks (mean of the vertical estimates), report their disagreement, and write the three files
vels = np.stack([r[0] for r in results.values()])
stds = np.stack([r[1] for r in results.values()])
tss = np.stack([r[2] for r in results.values()])
vel_final = np.nanmean(vels, 0)
std_final = np.sqrt(np.nanmean(stds ** 2, 0))
ts_final = np.nanmean(tss, 0)
first_valid = np.nanargmax(np.isfinite(ts_final).any(axis=(1, 2)))
ts_final = ts_final - ts_final[first_valid]                       # zero at the first epoch that has data
if len(results) == 2:
    diff = vels[0] - vels[1]
    print("asc - desc vertical velocity: median %.2f, robust SD %.2f mm/yr  (large values = horizontal motion or processing problems)" % (
        np.nanmedian(diff), 1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff)))))
write_stack(INSAR / "velocity_vertical_mm_yr.tif", vel_final)
write_stack(INSAR / "velocity_std_mm_yr.tif", std_final)
write_stack(INSAR / "disp_vertical_monthly_mm.tif", ts_final, names=MONTHS)
print("valid velocity share: %.1f%% of grid" % (100 * np.isfinite(vel_final).mean()))

# %%
# Cell 12 - Look at it
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
im = ax[0].imshow(vel_final, cmap="RdBu", vmin=-30, vmax=30); ax[0].set_title("Vertical velocity (mm/yr)"); plt.colorbar(im, ax=ax[0], shrink=0.8)
im = ax[1].imshow(std_final, cmap="magma"); ax[1].set_title("Velocity std (mm/yr)"); plt.colorbar(im, ax=ax[1], shrink=0.8)
ax[2].plot(np.nanmean(ts_final, axis=(1, 2))); ax[2].set_title("Mean displacement (mm)")
ax[2].set_xticks(range(0, len(MONTHS), 12)); ax[2].set_xticklabels(YEARS, rotation=45)
for a in ax[:2]:
    a.axis("off")
fig.savefig(OUT / "figures" / "04_insar_overview.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Before you move on, sanity-check:** (1) velocity over presumed-stable ground (hills away from the lake) should be about 0 +/- 2-3 mm/yr;
# (2) decorrelated areas (greenhouses, water) should be empty, not noisy; (3) a smooth, large-scale ramp across the whole scene means
# the reference point or orbit error is wrong. Notebook 07 repeats these checks quantitatively.
#
# **Next:** `05_build_features`.
