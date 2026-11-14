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

# @@boot_env@@

# @@boot_cfg@@

# @@gee_init@@

# @@gee_helpers@@

# @@io_helpers@@

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
