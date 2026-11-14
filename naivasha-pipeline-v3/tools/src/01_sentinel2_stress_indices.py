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

# @@boot_env@@

# @@boot_cfg@@

# @@gee_init@@

# @@gee_helpers@@

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
