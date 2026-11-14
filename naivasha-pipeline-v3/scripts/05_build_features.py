# %% [markdown]
# # 05 - Build the analysis tables
# Turns the rasters from notebooks 01-04 (or from the demo generator) into two tables for machine learning:
#
# | Table | One row per | Used for |
# |---|---|---|
# | `static_pixels.parquet` | land pixel | subsidence model (target = InSAR velocity) and the final susceptibility map |
# | `panel_stress.parquet`  | sampled pixel x forecast month | stress forecast model (target = stress score `H` months ahead) |
#
# **Stress score** (per pixel, per month): the mean of the standardised anomalies `-z(NDVI)`, `-z(NDMI)`, `+z(MSI)`, where each
# z-score compares a month with the same calendar month of the baseline years. Positive = more stress than normal.
#
# Steps: load -> mask -> gap-fill -> anomalies -> stress score -> trends -> distances -> join InSAR -> spatial blocks -> save.

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
# Cell 4 - Settings and imports
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)     # all-NaN slices are expected over water / masked areas
ensure("scipy", "pyarrow", "pyproj")
from itertools import combinations
from scipy import ndimage
from scipy.stats import norm
from scipy.spatial import cKDTree
from pyproj import Transformer

T, NY = len(MONTHS), len(YEARS)
HORIZON = CFG["forecast_horizon"]
B0 = CFG["baseline_years"][0] - YEARS[0]               # baseline window as year indices
B1 = CFG["baseline_years"][1] - YEARS[0] + 1
TREND_FROM = 1                                         # skip the first (partial) Sentinel-2 year in trends
AG_MIN_FRAC = 0.5                                      # a pixel is "agricultural" if cropland + grassland >= 50 %
DER = DATA / "derived"
DER.mkdir(exist_ok=True)
rng = np.random.default_rng(CFG["seed"])
print("months:", T, "| forecast horizon:", HORIZON, "months | baseline years index", B0, "to", B1 - 1)

# %% [markdown]
# ## 1. Static layers and masks

# %%
# Cell 5 - Load static rasters
elev = read_stack(DATA / "static" / "elevation_m.tif")[0][0]
slope = read_stack(DATA / "static" / "slope_deg.tif")[0][0]
clay = read_stack(DATA / "static" / "clay_pct.tif")[0][0]
upa = read_stack(DATA / "static" / "upa_km2.tif")[0][0]
lake_occ = read_stack(DATA / "static" / "lake_occurrence.tif")[0][0]
lc_arr, lc_names = read_stack(DATA / "static" / "landcover_fractions.tif")
LC = dict(zip(lc_names, lc_arr))
print("land-cover bands:", lc_names)

# %%
# Cell 6 - Masks: lake, land, agricultural land
lake = np.nan_to_num(lake_occ, nan=0) >= 50
land = (~lake) & np.isfinite(elev)
ag = land & ((LC["crop"] + LC["grass"]) >= AG_MIN_FRAC)
pix = np.flatnonzero(land.ravel())          # flat index of every land pixel
N = pix.size
print(f"land pixels: {N:,} | agricultural pixels: {int(ag.sum()):,} | lake pixels: {int(lake.sum()):,}")

# %%
# Cell 7 - Helpers to move between maps [rows, cols] and pixel vectors [N]
def to_vec(cube):
    """[..., H, W] -> [..., N] (land pixels only)."""
    return cube.reshape(cube.shape[:-2] + (H * W,))[..., pix]


def to_map(vec, fill=np.nan):
    """[..., N] -> [..., H, W]."""
    out = np.full(vec.shape[:-1] + (H * W,), fill, dtype="float32")
    out[..., pix] = vec
    return out.reshape(vec.shape[:-1] + (H, W))


def yearly(a):
    """[T, N] -> [years, 12, N]."""
    return a.reshape(NY, 12, -1)

print("helpers ready")

# %% [markdown]
# ## 2. Sentinel-2 indices -> gap-filled -> anomalies

# %%
# Cell 8 - Load NDVI, NDMI and MSI as [months, pixels]; fill short cloud gaps (max 2 months, inside the series)
def fill_gaps(a, limit=2):
    df = pd.DataFrame(a)
    return df.interpolate(limit=limit, limit_area="inside", axis=0).to_numpy(dtype="float32")


raw = {}
for name in ["NDVI", "NDMI", "MSI"]:
    raw[name] = fill_gaps(to_vec(read_years(DATA / "s2", name)))
    print(name, raw[name].shape, "valid share: %.2f" % np.isfinite(raw[name]).mean())

# %%
# Cell 9 - Standardised anomaly (z-score) against the monthly climatology of the baseline years
def monthly_z(a, floor=0.01):
    c = yearly(a)
    base = c[B0:B1]
    mu, sd = np.nanmean(base, 0), np.nanstd(base, 0)
    return ((c - mu) / np.maximum(sd, floor)).reshape(a.shape).astype("float32")


def vci(a):
    """Vegetation Condition Index 0-100 against the baseline min/max of each calendar month."""
    c = yearly(a)
    lo, hi = np.nanmin(c[B0:B1], 0), np.nanmax(c[B0:B1], 0)
    return (100 * (c - lo) / np.maximum(hi - lo, 1e-3)).reshape(a.shape).astype("float32")


z_ndvi, z_ndmi, z_msi = (monthly_z(raw[k]) for k in ["NDVI", "NDMI", "MSI"])
vci_ndvi = vci(raw["NDVI"])
print("z-scores ready")

# %%
# Cell 10 - STRESS SCORE: higher = drier / less green than normal for that calendar month
stress = np.nanmean(np.stack([-z_ndvi, -z_ndmi, z_msi]), axis=0).astype("float32")      # [T, N]
b_low, b_high = CFG["stress_class_breaks"]
stress_class = np.where(np.isfinite(stress), np.digitize(stress, [b_low, b_high]), -1)   # 0 low, 1 moderate, 2 high
print("stress score: mean %.3f, sd %.3f | share high-stress (ag pixels): %.3f" % (
    np.nanmean(stress), np.nanstd(stress), np.nanmean(stress_class[:, ag.ravel()[pix]] == 2)))
np.save(DER / "stress_score.npy", to_map(stress))       # [T, H, W] for maps and time-lapses
print("saved stress_score.npy")

# %% [markdown]
# ## 3. Long-term statistics and trends per pixel

# %%
# Cell 11 - Theil-Sen slope and Mann-Kendall p-value for annual series
def theil_sen_mk(Y):
    """Y [n_years, N] -> (Theil-Sen slope per year, Mann-Kendall two-sided p). NaN if < 6 valid years."""
    n = Y.shape[0]
    slopes = np.stack([(Y[j] - Y[i]) / (j - i) for i, j in combinations(range(n), 2)]).astype("float32")
    ts = np.nanmedian(slopes, 0)
    s = np.nansum(np.sign(slopes), 0)
    z = (s - np.sign(s)) / np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
    p = 2 * (1 - norm.cdf(np.abs(z)))
    bad = np.isfinite(Y).sum(0) < 6
    ts[bad], p[bad] = np.nan, np.nan
    return ts, p


def annual_mean(a):
    return np.nanmean(yearly(a), axis=1)[TREND_FROM:]


feat = {}                                  # pixel-level features (each is a vector of length N)
for name in ["NDVI", "NDMI"]:
    ann = annual_mean(raw[name])
    feat[f"{name.lower()}_mean"] = np.nanmean(raw[name], 0)
    feat[f"{name.lower()}_trend"], feat[f"{name.lower()}_trend_p"] = theil_sen_mk(ann)
feat["ndvi_sd"] = np.nanstd(raw["NDVI"], 0)
feat["stress_mean"] = np.nanmean(stress, 0)
feat["stress_trend"], _ = theil_sen_mk(annual_mean(stress))
feat["stress_frac_high"] = np.nanmean(np.where(stress_class == 2, 1.0, np.where(stress_class < 0, np.nan, 0.0)), 0)
print({k: round(float(np.nanmean(v)), 4) for k, v in feat.items()})

# %%
# Cell 12 - Streaming annual climate statistics (rain and ET sums, temperature mean) and their trends
def annual_climate(name, how):
    out = []
    for y in YEARS:
        a = read_stack(DATA / "climate" / f"{name}_{y}.tif")[0]
        out.append(np.nansum(a, 0) if how == "sum" else np.nanmean(a, 0))
    return to_vec(np.stack(out))                      # [years, N]


for var, how, label in [("rain_mm", "sum", "rain"), ("et_mm", "sum", "et"), ("temp_c", "mean", "temp")]:
    ann = annual_climate(var, how)
    feat[f"{label}_mean"] = np.nanmean(ann, 0)
    feat[f"{label}_trend"], _ = theil_sen_mk(ann[TREND_FROM:])
print("climate features:", [k for k in feat if k.startswith(("rain", "et_", "temp"))])

# %% [markdown]
# ## 4. Distances, boreholes and (optional) geology

# %%
# Cell 13 - Pixel coordinates, distance to the lake and to points of interest (km)
to_utm = Transformer.from_crs("EPSG:4326", GRID["crs"], always_xy=True)
rows_, cols_ = np.divmod(pix, W)
x = GRID["xmin"] + (cols_ + 0.5) * GRID["res"]
y = GRID["ymax"] - (rows_ + 0.5) * GRID["res"]
dist_lake = (ndimage.distance_transform_edt(~lake) * GRID["res"] / 1000).ravel()[pix]
dist = {"dist_lake_km": dist_lake}
for name, (lon, lat) in CFG["points"].items():
    px, py = to_utm.transform(lon, lat)
    dist[f"dist_{name}_km"] = np.hypot(x - px, y - py) / 1000
print({k: round(float(v.mean()), 1) for k, v in dist.items()})

# %%
# Cell 14 - OPTIONAL: boreholes (CSV with lon, lat) -> distance to nearest borehole and borehole count within 5 km
from shapely.geometry import Point, shape

bh_file = DATA / "ancillary" / "boreholes.csv"
if bh_file.exists():
    bh = pd.read_csv(bh_file)
    aoi_wgs84 = shape(CFG["aoi"]["geometry"])
    inside_aoi = bh.apply(lambda row: aoi_wgs84.covers(Point(row["lon"], row["lat"])), axis=1)
    bh = bh.loc[inside_aoi]
    if bh.empty:
        print("No boreholes fall inside the AOI - skipping the optional borehole features.")
    else:
        bx, by = to_utm.transform(bh["lon"].to_numpy(), bh["lat"].to_numpy())
        tree = cKDTree(np.c_[bx, by])
        dist["dist_borehole_km"] = tree.query(np.c_[x, y])[0] / 1000
        dist["borehole_n_5km"] = tree.query_ball_point(np.c_[x, y], r=5000, return_length=True).astype("float32")
        print("boreholes inside AOI used:", len(bh))
else:
    print("No boreholes.csv found in data/ancillary - skipping (optional; get abstraction points from WRA).")

# %%
# Cell 15 - OPTIONAL: geology polygons (GeoPackage with a text column 'unit') -> integer code per pixel
geo_file = DATA / "ancillary" / "geology.gpkg"
geology = {}
if geo_file.exists():
    ensure("geopandas")
    import geopandas as gpd
    from rasterio import features
    g = gpd.read_file(geo_file).to_crs(GRID["crs"])
    codes = {u: i + 1 for i, u in enumerate(sorted(g["unit"].astype(str).unique()))}
    shapes = [(geom, codes[str(u)]) for geom, u in zip(g.geometry, g["unit"])]
    rast = features.rasterize(shapes, out_shape=(H, W), transform=TRANSFORM, fill=0, dtype="int16")
    geology["geology_code"] = rast.ravel()[pix].astype("float32")
    print("geology units:", codes)
else:
    print("No geology.gpkg found - skipping (optional; lithology helps explain compressible sediments).")

# %% [markdown]
# ## 5. Join InSAR and assemble the pixel table

# %%
# Cell 16 - InSAR velocity (mm/yr, negative = sinking) and its uncertainty
vel = to_vec(read_stack(DATA / "insar" / "velocity_vertical_mm_yr.tif")[0][0])
std_file = DATA / "insar" / "velocity_std_mm_yr.tif"
vel_std = to_vec(read_stack(std_file)[0][0]) if std_file.exists() else np.full(N, np.nan, dtype="float32")
print("pixels with valid InSAR velocity: %.1f%% of land" % (100 * np.isfinite(vel).mean()))

# %%
# Cell 17 - Spatial blocks for honest cross-validation (neighbouring pixels share a block id)
bs = max(int(round(CFG["block_km"] * 1000 / GRID["res"])), 1)
n_bcols = int(np.ceil(W / bs))
block = ((rows_ // bs) * n_bcols + (cols_ // bs)).astype("int32")
print(f"block size {bs} px ({CFG['block_km']} km) -> {np.unique(block).size} blocks")

# %%
# Cell 18 - Build and save the static pixel table
static = pd.DataFrame({"pix": pix, "row": rows_, "col": cols_, "x": x, "y": y, "block": block,
                       "vel": vel, "vel_std": vel_std,
                       "elevation": elev.ravel()[pix], "slope": slope.ravel()[pix], "clay": clay.ravel()[pix],
                       "upa_log": np.log10(np.clip(upa.ravel()[pix], 1e-3, None)),
                       "crop_frac": LC["crop"].ravel()[pix], "grass_frac": LC["grass"].ravel()[pix],
                       "tree_frac": LC["tree"].ravel()[pix], "built_frac": LC["built"].ravel()[pix],
                       "bare_frac": LC["bare"].ravel()[pix], "is_ag": ag.ravel()[pix]})
for k, v in {**dist, **geology, **feat}.items():
    static[k] = v
static.to_parquet(DER / "static_pixels.parquet", index=False)
print(static.shape, "->", DER / "static_pixels.parquet")

# %% [markdown]
# ## 6. Pixel-by-month panel for the stress forecast
# Each row = "standing at month *t* in pixel *p*, what will the stress score be at *t + H*?" Only information available
# at *t* is used as input (no future climate), so the test is a genuine forecast.

# %%
# Cell 19 - Choose sample pixels (agricultural, valid), stratified by spatial block
cand = static[static["is_ag"] & np.isfinite(static["stress_mean"])]
frac = min(1.0, CFG["sample_pixels"] / len(cand))
shuffled = cand.sample(frac=1.0, random_state=CFG["seed"])
rank_in_block = shuffled.groupby("block").cumcount()
keep_n = np.maximum(1, np.round(shuffled.groupby("block")["pix"].transform("size") * frac)).astype(int)
sample = shuffled[rank_in_block < keep_n].sort_values("pix")
pos = np.searchsorted(pix, sample["pix"].to_numpy())          # positions inside the [.., N] arrays
print(f"{len(sample):,} sample pixels in {sample['block'].nunique()} blocks (of {len(cand):,} candidates)")

# %%
# Cell 20 - Climate anomalies at the sample pixels: monthly rain, 3- and 6-month rain, temperature, PET, ET
def climate_sample(name):
    return to_vec(read_years(DATA / "climate", name))[:, pos]


def rolling_sum(a, k):
    return pd.DataFrame(a).rolling(k, min_periods=k).sum().to_numpy(dtype="float32")


rain = climate_sample("rain_mm")
dyn = {
    "stress": stress[:, pos], "z_ndvi": z_ndvi[:, pos], "z_ndmi": z_ndmi[:, pos], "z_msi": z_msi[:, pos],
    "rain1_z": monthly_z(rain, floor=1.0), "rain3_z": monthly_z(rolling_sum(rain, 3), floor=1.0),
    "rain6_z": monthly_z(rolling_sum(rain, 6), floor=1.0),
    "temp_z": monthly_z(climate_sample("temp_c"), floor=0.1),
    "pet_z": monthly_z(climate_sample("pet_mm"), floor=1.0),
    "et_z": monthly_z(climate_sample("et_mm"), floor=1.0),
}
print({k: v.shape for k, v in list(dyn.items())[:3]}, "...")

# %%
# Cell 21 - Assemble the panel (features at month t, target = stress at t + HORIZON)
origins = np.arange(3, T - HORIZON)                       # need 3 months of history for lags
S = len(pos)
cols = {"pix": np.tile(sample["pix"].to_numpy(), len(origins)),
        "block": np.tile(sample["block"].to_numpy(), len(origins)),
        "origin": np.repeat(origins, S), "target_idx": np.repeat(origins + HORIZON, S)}
for k, v in dyn.items():
    cols[k] = v[origins].ravel()
cols["stress_lag1"] = dyn["stress"][origins - 1].ravel()
cols["stress_lag2"] = dyn["stress"][origins - 2].ravel()
cols["rain3_z_lag3"] = dyn["rain3_z"][origins - 3].ravel()
tmonth = (origins + HORIZON) % 12                          # calendar month of the TARGET (known in advance)
cols["month_sin"] = np.repeat(np.sin(2 * np.pi * tmonth / 12), S)
cols["month_cos"] = np.repeat(np.cos(2 * np.pi * tmonth / 12), S)
cols["target"] = dyn["stress"][origins + HORIZON].ravel()
panel = pd.DataFrame(cols)
panel["target_year"] = YEARS[0] + panel["target_idx"] // 12
static_cols = [c for c in static.columns if c not in ("pix", "row", "col", "x", "y", "block", "is_ag")]
panel = panel.merge(static[["pix"] + static_cols], on="pix", how="left")
panel = panel[np.isfinite(panel["target"])].reset_index(drop=True)
print(panel.shape, "| target years", panel["target_year"].min(), "-", panel["target_year"].max())

# %%
# Cell 22 - Save the panel and a small metadata file that notebook 06 reads
panel.to_parquet(DER / "panel_stress.parquet", index=False)
dynamic_features = ["stress", "stress_lag1", "stress_lag2", "z_ndvi", "z_ndmi", "z_msi", "rain1_z", "rain3_z", "rain6_z",
                    "rain3_z_lag3", "temp_z", "pet_z", "et_z", "month_sin", "month_cos"]
driver_features = [c for c in ["elevation", "slope", "clay", "upa_log", "crop_frac", "grass_frac", "tree_frac", "built_frac",
                               "bare_frac", "borehole_n_5km", "geology_code"] if c in static.columns]
driver_features += [c for c in static.columns if c.startswith("dist_")]        # distances to lake, points, boreholes
veg_features = ["ndvi_mean", "ndvi_sd", "ndvi_trend", "ndmi_mean", "ndmi_trend", "stress_mean", "stress_trend",
                "stress_frac_high", "rain_mean", "rain_trend", "et_mean", "et_trend", "temp_mean", "temp_trend"]
meta = dict(dynamic_features=dynamic_features, driver_features=driver_features, veg_features=veg_features,
            horizon=HORIZON, n_panel=len(panel), n_static=len(static), n_sample_pixels=int(S))
(DER / "feature_meta.json").write_text(json.dumps(meta, indent=2))
print("saved panel_stress.parquet and feature_meta.json")

# %% [markdown]
# ## 7. Quick look

# %%
# Cell 23 - Mean stress map, InSAR velocity map and area-mean stress vs rainfall
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
im = ax[0].imshow(to_map(feat["stress_mean"]), cmap="RdYlBu_r", vmin=-0.5, vmax=0.5); ax[0].set_title("Mean stress score 2015-2024")
plt.colorbar(im, ax=ax[0], shrink=0.7)
im = ax[1].imshow(to_map(vel), cmap="RdBu", vmin=-30, vmax=30); ax[1].set_title("InSAR vertical velocity (mm/yr)")
plt.colorbar(im, ax=ax[1], shrink=0.7)
for a in ax[:2]:
    a.axis("off")
ag_pos = np.flatnonzero(static["is_ag"].to_numpy())
ax[2].plot(np.nanmean(stress[:, ag_pos], 1), label="mean stress (ag pixels)")
ax[2].plot(np.nanmean(dyn["rain3_z"], 1), alpha=0.6, label="3-month rain z (sample)")
ax[2].set_xticks(range(0, T, 12)); ax[2].set_xticklabels(YEARS, rotation=45); ax[2].legend(); ax[2].set_title("Stress vs rainfall anomaly")
fig.savefig(OUT / "figures" / "05_overview.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Next:** `06_random_forest_models`.
