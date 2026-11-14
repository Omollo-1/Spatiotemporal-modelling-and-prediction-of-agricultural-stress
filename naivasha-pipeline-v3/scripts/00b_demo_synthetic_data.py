# %% [markdown]
# # 00b - Synthetic demo data (DEMO mode only)
# Creates a **synthetic** Naivasha-like dataset with exactly the same file names and formats that notebooks 01-04 produce
# from real satellite data. This lets you run and test notebooks 05-08 end to end on any laptop or on Colab with no
# accounts, no downloads and no waiting.
#
# > **Important:** the numbers are invented. The coupling between subsidence and crop stress is built in on purpose so the
# > pipeline has something to find. Never quote demo outputs as scientific results.
#
# Skip this notebook in `real` mode.

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
# Cell 4 - Safety check
if MODE != "demo":
    raise SystemExit("Config is in REAL mode. Set MODE = 'demo' in notebook 00 to build synthetic data.")
ensure("scipy", "pyproj")
from scipy.ndimage import gaussian_filter, distance_transform_edt
from pyproj import Transformer
rng = np.random.default_rng(CFG["seed"])
T = len(MONTHS)
print("Grid", W, "x", H, "| months", T)

# %% [markdown]
# ## 1. Coordinates and helpers

# %%
# Cell 5 - Pixel coordinates (metres) and small helper functions
to_utm = Transformer.from_crs("EPSG:4326", GRID["crs"], always_xy=True)
col, row = np.meshgrid(np.arange(W), np.arange(H))
X = GRID["xmin"] + (col + 0.5) * GRID["res"]
Y = GRID["ymax"] - (row + 0.5) * GRID["res"]


def dist_km(lon, lat):
    x0, y0 = to_utm.transform(lon, lat)
    return np.hypot(X - x0, Y - y0) / 1000


def smooth_noise(km, scale=1.0):
    n = gaussian_filter(rng.normal(size=(H, W)), max(km * 1000 / GRID["res"], 0.5))
    return scale * n / n.std()


def gauss(d_km, sigma_km):
    return np.exp(-0.5 * (d_km / sigma_km) ** 2)


d_olk = dist_km(*CFG["points"]["olkaria_geothermal"])
d_long = dist_km(*CFG["points"]["mt_longonot"])
d_town = dist_km(*CFG["points"]["naivasha_town"])
print("helpers ready")

# %% [markdown]
# ## 2. Static layers: lake, terrain, soil, land cover

# %%
# Cell 6 - Lake, elevation, slope, clay, upstream area
x0, y0 = to_utm.transform(36.355, -0.77)
lake = (((X - x0) / 9000) ** 2 + ((Y - y0) / 6500) ** 2) < 1
lake_occ = 100 * gaussian_filter(lake.astype("float32"), 0.6)
d_lake = distance_transform_edt(~lake) * GRID["res"] / 1000
elev = 1890 + 10 * d_lake + 350 * gauss(d_long, 4) + smooth_noise(4, 40)
gy, gx = np.gradient(elev, GRID["res"])
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
clay = np.clip(35 + smooth_noise(5, 8), 10, 60)
upa = np.exp(2 + 0.3 * d_lake + smooth_noise(3, 1.0))
print("lake pixels:", int(lake.sum()), "| elevation range", int(elev.min()), "-", int(elev.max()), "m")

# %%
# Cell 7 - Land cover classes -> fractions
# classes: 0 crop, 1 grass, 2 tree, 3 shrub, 4 built, 5 bare, 6 water
cls = np.full((H, W), 1)
cls[(rng.random((H, W)) < 0.18)] = 3
cls[(d_lake > 0.5) & (d_lake < 9) & (elev < 2000) & (slope < 4) & (smooth_noise(3) > -0.6)] = 0
cls[(d_olk < 3) | (d_long < 2)] = 5
cls[(elev > 2150) | (elev > 2050) & (smooth_noise(2) > 0.5)] = 2
cls[d_town < 2.5] = 4
cls[lake] = 6
onehot = np.stack([gaussian_filter((cls == k).astype("float32"), 0.7) for k in range(7)])
fracs = onehot / onehot.sum(0, keepdims=True)
irrigated = (cls == 0) & (d_lake < 6)
print("cropland pixels:", int((cls == 0).sum()), "| irrigated:", int(irrigated.sum()))

# %% [markdown]
# ## 3. Climate (monthly rain, temperature, PET, ET) with drought spells

# %%
# Cell 8 - Rainfall anomaly series (droughts 2016-17 and 2020-22, wet late 2019)
month_idx = lambda y, m: (y - YEARS[0]) * 12 + (m - 1)
anom = np.zeros(T)
for t in range(1, T):
    anom[t] = 0.4 * anom[t - 1] + rng.normal(0, 0.2)
for (y0_, m0_, y1_, m1_, v) in [(2016, 10, 2017, 3, -0.6), (2019, 10, 2019, 12, 0.5),
                                (2020, 10, 2021, 3, -0.4), (2021, 10, 2022, 12, -0.45)]:
    anom[month_idx(y0_, m0_):month_idx(y1_, m1_) + 1] += v
clim = np.array([60, 50, 100, 170, 120, 40, 30, 35, 35, 70, 110, 90], dtype="float32")   # bimodal rainfall (mm/month)
cal_month = np.arange(T) % 12
elev_factor = (1 + (elev - 1900) / 1500) * (1 + 0.1 * smooth_noise(6))
rain = np.clip(clim[cal_month][:, None, None] * (1 + anom)[:, None, None] * elev_factor, 0, None).astype("float32")
rain *= rng.lognormal(0, 0.1, size=rain.shape).astype("float32")
temp = (19 - 0.0065 * (elev - 1900))[None] + (1.5 * np.cos(2 * np.pi * (cal_month - 1) / 12))[:, None, None] + rng.normal(0, 0.3, (T, H, W))
pet = 120 + 6 * (temp - 19) + 10 * np.cos(2 * np.pi * (cal_month - 1) / 12)[:, None, None]
sm = np.zeros_like(rain)                                   # simple soil-moisture memory
sm[0] = rain[0]
for t in range(1, T):
    sm[t] = 0.55 * sm[t - 1] + 0.45 * rain[t]
sm_anom = (sm - sm.mean(0)) / (sm.std(0) + 1e-6)
et = np.minimum(0.9 * pet, 0.6 * sm + 60 * irrigated[None])
print("annual rain (mm) at lake-side farms:", np.round(rain[:, irrigated].mean(1).reshape(-1, 12).sum(1)).astype(int))

# %% [markdown]
# ## 4. Land subsidence (the hidden "truth" + what InSAR would measure)

# %%
# Cell 9 - Subsidence velocity field: farm abstraction bowls + Olkaria bowl + small Longonot uplift
farm_centres = [(36.33, -0.82), (36.41, -0.79), (36.28, -0.97)]
A = np.clip(sum(gauss(dist_km(lo, la), 3.5) for lo, la in farm_centres), 0, 1)      # abstraction intensity 0-1
vel_true = (-18 * A - 30 * gauss(d_olk, 2.5) + 6 * gauss(d_long, 4)
            + 1.0 * (X - X.mean()) / np.ptp(X) + smooth_noise(4, 0.8))                 # mm/yr, negative = sinking
print("true velocity range (mm/yr): %.1f to %.1f" % (vel_true.min(), vel_true.max()))

# %%
# Cell 10 - Monthly vertical displacement series (starts Jan 2016, like usable Sentinel-1 coverage) and measured velocity
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices are expected
t0 = month_idx(2016, 1)
disp = np.full((T, H, W), np.nan, dtype="float32")
for t in range(t0, T):
    disp[t] = vel_true / 12 * (t - t0) - 1.5 * sm_anom[t] * A + rng.normal(0, 1.5, (H, W))   # elastic seasonal term + noise
decorr = lake | (irrigated & (rng.random((H, W)) < 0.25))     # greenhouses and water decorrelate
disp[:, decorr] = np.nan
tt = np.arange(t0, T) / 12.0
D = disp[t0:]
tm = tt.mean()
vel_meas = ((D - np.nanmean(D, 0)) * (tt - tm)[:, None, None]).sum(0) / ((tt - tm) ** 2).sum()
resid = D - (np.nanmean(D, 0) + vel_meas * (tt - tm)[:, None, None])
vel_std = np.sqrt(np.nanmean(resid ** 2, 0) / ((tt - tm) ** 2).sum())
print("measured velocity (mm/yr): median %.2f, min %.1f" % (np.nanmedian(vel_meas), np.nanmin(vel_meas)))

# %% [markdown]
# ## 5. Sentinel-2 style vegetation indices (with cloud gaps and no data before mid-2015)

# %%
# Cell 11 - NDVI, NDMI, MSI, NDRE, EVI
base = np.array([0.55, 0.40, 0.70, 0.35, 0.20, 0.12, -0.10])           # NDVI by class
sens = np.array([0.07, 0.09, 0.03, 0.08, 0.02, 0.03, 0.0])             # response to soil moisture
ndvi_base = np.tensordot(base, fracs, axes=1)
ndvi_sens = np.tensordot(sens, fracs, axes=1)
ndvi_sens = np.where(irrigated, 0.025, ndvi_sens)                      # irrigated crops buffer climate
chronic = 0.25 * A * fracs[0] * (np.arange(T) / T)[:, None, None]      # chronic decline where abstraction is high
state = np.zeros(T)                                                    # persistent vegetation condition (not explained by rain)
for t in range(1, T):
    state[t] = 0.8 * state[t - 1] + rng.normal()
state /= state.std()
amp = 0.035 * (0.6 + 0.8 * np.clip(smooth_noise(8, 0.5) + 0.5, 0, 1)) * (1 - 0.5 * irrigated)
ndvi = ndvi_base + ndvi_sens * sm_anom - chronic + amp * state[:, None, None] + rng.normal(0, 0.02, (T, H, W))
ndvi = np.clip(ndvi, -0.2, 0.95).astype("float32")
ndmi = (0.8 * ndvi - 0.25 + rng.normal(0, 0.015, (T, H, W))).astype("float32")
msi = ((1 - ndmi) / (1 + ndmi)).astype("float32")
ndre = (0.55 * ndvi - 0.05 + rng.normal(0, 0.01, (T, H, W))).astype("float32")
evi = (0.75 * ndvi + rng.normal(0, 0.015, (T, H, W))).astype("float32")
p_cloud = 0.02 + 0.10 * np.clip(rain / 200, 0, 1)
cloud = rng.random((T, H, W)) < p_cloud
cloud[:6] = True                                                       # Sentinel-2 starts mid-2015
indices = {"NDVI": ndvi, "NDMI": ndmi, "MSI": msi, "NDRE": ndre, "EVI": evi}
for k in indices:
    indices[k] = np.where(cloud, np.nan, indices[k]).astype("float32")
print("share of missing months:", round(float(cloud.mean()), 3))

# %% [markdown]
# ## 6. Write everything in the same layout as the real pipeline

# %%
# Cell 12 - Sentinel-2 indices and climate, one GeoTIFF per variable-year (12 monthly bands)
for name, cube in {**indices, "rain_mm": rain, "temp_c": temp, "pet_mm": pet, "et_mm": et}.items():
    folder = DATA / ("s2" if name in indices else "climate")
    for i, y in enumerate(YEARS):
        write_stack(folder / f"{name}_{y}.tif", cube[i * 12:(i + 1) * 12], names=MONTHS[i * 12:(i + 1) * 12])
print("wrote", len(list((DATA / "s2").glob("*.tif"))), "S2 files and", len(list((DATA / "climate").glob("*.tif"))), "climate files")

# %%
# Cell 13 - Static layers and InSAR products
write_stack(DATA / "static" / "elevation_m.tif", elev)
write_stack(DATA / "static" / "slope_deg.tif", slope)
write_stack(DATA / "static" / "clay_pct.tif", clay)
write_stack(DATA / "static" / "upa_km2.tif", upa)
write_stack(DATA / "static" / "lake_occurrence.tif", lake_occ)
write_stack(DATA / "static" / "landcover_fractions.tif", fracs, names=["crop", "grass", "tree", "shrub", "built", "bare", "water"])
write_stack(DATA / "insar" / "velocity_vertical_mm_yr.tif", vel_meas)
write_stack(DATA / "insar" / "velocity_std_mm_yr.tif", vel_std)
write_stack(DATA / "insar" / "disp_vertical_monthly_mm.tif", disp, names=MONTHS)
print("static + InSAR written")

# %%
# Cell 14 - Optional ancillary tables: boreholes and GNSS/levelling points (synthetic)
inv = Transformer.from_crs(GRID["crs"], "EPSG:4326", always_xy=True)
idx = np.argwhere(irrigated)
pick = idx[rng.choice(len(idx), 25, replace=False)]
lon, lat = inv.transform(X[pick[:, 0], pick[:, 1]], Y[pick[:, 0], pick[:, 1]])
pd.DataFrame({"lon": lon, "lat": lat, "abstraction_m3_day": rng.integers(50, 800, 25)}).to_csv(DATA / "ancillary" / "boreholes.csv", index=False)
ok = np.argwhere(np.isfinite(vel_meas) & (A > 0.2))
pick = ok[rng.choice(len(ok), 6, replace=False)]
lon, lat = inv.transform(X[pick[:, 0], pick[:, 1]], Y[pick[:, 0], pick[:, 1]])
pd.DataFrame({"lon": lon, "lat": lat, "vel_up_mm_yr": vel_true[pick[:, 0], pick[:, 1]] + rng.normal(0, 1.0, 6)}).to_csv(DATA / "ancillary" / "gnss.csv", index=False)
(DATA / "README_SYNTHETIC.txt").write_text("These files are SYNTHETIC demo data generated by 00b_demo_synthetic_data. Not real observations.\n")
print("ancillary tables written")

# %% [markdown]
# ## 7. Look at it

# %%
# Cell 15 - Sanity plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, (arr, title, cmap, kw) in zip(ax, [(elev, "Elevation (m)", "terrain", {}),
                                          (cls, "Land cover class (0 = crop)", "tab10", {}),
                                          (vel_meas, "InSAR velocity (mm/yr)", "RdBu", dict(vmin=-30, vmax=30)),
                                          (np.nanmean(ndvi, 0), "Mean NDVI", "YlGn", {})]):
    im = a.imshow(arr, cmap=cmap, **kw); a.set_title(title); a.axis("off"); plt.colorbar(im, ax=a, shrink=0.7)
fig.suptitle("SYNTHETIC DEMO DATA", color="red")
fig.savefig(OUT / "figures" / "00b_demo_overview.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Next:** run `05_build_features`.
