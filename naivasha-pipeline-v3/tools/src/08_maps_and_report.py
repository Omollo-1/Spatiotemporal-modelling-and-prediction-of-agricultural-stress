# %% [markdown]
# # 08 - Maps, time-lapse, web map and results summary
# Turns the model outputs into things you can show: publication-style figures, a stress time-lapse GIF, Cloud-Optimised GeoTIFFs,
# an interactive web map (open the HTML in any browser) and an auto-written `RESULTS_SUMMARY.md` that pulls the real numbers
# from notebooks 06 and 07, so the write-up can never drift from the analysis.

# @@boot_env@@

# @@boot_cfg@@

# @@io_helpers@@

# %%
# Cell 4 - Imports and inputs
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
ensure("matplotlib", "pillow", "folium", "rasterio", "pyproj")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw
import folium
from rasterio.warp import calculate_default_transform, reproject, Resampling

DER, TAB, FIG, MAPS = DATA / "derived", OUT / "tables", OUT / "figures", OUT / "maps"
required_inputs = [
    DER / "static_pixels.parquet",
    DER / "stress_score.npy",
    TAB / "06_metrics.json",
    TAB / "07_results.json",
    DATA / "static" / "lake_occurrence.tif",
    DATA / "insar" / "velocity_vertical_mm_yr.tif",
    MAPS / "pred_velocity_mm_yr.tif",
    MAPS / "pred_p_severe.tif",
    MAPS / "hotspot_classes.tif",
]
missing_inputs = [path for path in required_inputs if not path.exists()]
if missing_inputs:
    missing_list = "\n".join(f"  - {path}" for path in missing_inputs)
    raise FileNotFoundError(
        "Notebook 08 inputs are missing. Run notebooks 01-04 (or generate the demo data), "
        "then 05, 06, and 07 in order before running 08.\n"
        f"Missing files:\n{missing_list}"
    )
static = pd.read_parquet(DER / "static_pixels.parquet")
stress_cube = np.load(DER / "stress_score.npy")
m06 = json.loads((TAB / "06_metrics.json").read_text())
m07 = json.loads((TAB / "07_results.json").read_text())
lake = np.nan_to_num(read_stack(DATA / "static" / "lake_occurrence.tif")[0][0], nan=0) >= 50
vel_obs = read_stack(DATA / "insar" / "velocity_vertical_mm_yr.tif")[0][0]
vel_pred = read_stack(MAPS / "pred_velocity_mm_yr.tif")[0][0]
p_severe = read_stack(MAPS / "pred_p_severe.tif")[0][0]
hot = read_stack(MAPS / "hotspot_classes.tif")[0][0]
stress_mean = np.full(H * W, np.nan, "float32"); stress_mean[static["pix"]] = static["stress_mean"]; stress_mean = stress_mean.reshape(H, W)
EXTENT_KM = [GRID["xmin"] / 1000, GRID["xmax"] / 1000, GRID["ymin"] / 1000, GRID["ymax"] / 1000]
print("inputs loaded")

# %% [markdown]
# ## 1. Publication-style maps

# %%
# Cell 5 - Map helper (lake outline, UTM km axes, colour bar)
def show(ax, arr, title, cmap, vmin=None, vmax=None, label=""):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=EXTENT_KM, interpolation="nearest")
    ax.contour(lake, levels=[0.5], colors="k", linewidths=0.6, extent=[EXTENT_KM[0], EXTENT_KM[1], EXTENT_KM[3], EXTENT_KM[2]])
    ax.set_title(title, fontsize=11); ax.set_xlabel("Easting (km, UTM 37S)"); ax.set_ylabel("Northing (km)")
    cb = plt.colorbar(im, ax=ax, shrink=0.8); cb.set_label(label)
    if MODE == "demo":
        ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, ha="center", va="center", fontsize=28, color="red", alpha=0.25, rotation=30)
    return im

print("ready")

# %%
# Cell 6 - Four-panel overview: observed velocity, predicted velocity, severe-subsidence probability, mean stress
fig, ax = plt.subplots(2, 2, figsize=(13, 12))
show(ax[0, 0], vel_obs, "InSAR vertical velocity (observed)", "RdBu", -30, 30, "mm/yr (negative = sinking)")
show(ax[0, 1], vel_pred, "Random Forest velocity (all land pixels)", "RdBu", -30, 30, "mm/yr")
show(ax[1, 0], p_severe, "Probability of severe subsidence", "OrRd", 0, 1, "probability")
show(ax[1, 1], stress_mean, "Mean agricultural stress score 2015-2024", "RdYlBu_r", -0.6, 0.6, "z-score (positive = stressed)")
fig.tight_layout()
fig.savefig(FIG / "08_overview_maps.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# Cell 7 - Hotspot map
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(7.5, 7))
im = show(ax, hot, "Hotspots: subsidence x agricultural stress", ListedColormap(["#eeeeee", "#4575b4", "#fdae61", "#d73027"]), -0.5, 3.5)
im.colorbar.set_ticks([0, 1, 2, 3]); im.colorbar.set_ticklabels(["neither", "subsiding", "stressed", "both"])
fig.savefig(FIG / "08_hotspots.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Stress time-lapse (GIF)

# %%
# Cell 8 - One frame per month, coloured with the same stress scale
cmap = plt.get_cmap("RdYlBu_r")
SCALE = max(1, 400 // max(H, W))
frames = []
for t in range(len(MONTHS)):
    a = np.clip((stress_cube[t] + 2) / 4, 0, 1)                      # -2..+2 -> 0..1
    rgba = (cmap(a) * 255).astype("uint8")
    rgba[~np.isfinite(stress_cube[t])] = [230, 230, 230, 255]
    rgba[lake] = [120, 170, 220, 255]
    img = Image.fromarray(rgba[..., :3]).resize((W * SCALE, H * SCALE), Image.NEAREST)
    ImageDraw.Draw(img).text((6, 6), MONTHS[t] + (" SYNTHETIC" if MODE == "demo" else ""), fill=(0, 0, 0))
    frames.append(img)
frames[0].save(MAPS / "stress_timelapse.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
print("saved", MAPS / "stress_timelapse.gif", f"({len(frames)} frames)")

# %% [markdown]
# ## 3. Cloud-Optimised GeoTIFFs (for QGIS, ArcGIS, web apps)

# %%
# Cell 9 - Convert every output raster to COG (falls back to a plain copy if the GDAL COG driver is missing)
import shutil
from rasterio.shutil import copy as rio_copy
cog_dir = MAPS / "cog"
cog_dir.mkdir(exist_ok=True)
for tif in sorted(MAPS.glob("*.tif")):
    try:
        rio_copy(tif, cog_dir / tif.name, driver="COG", compress="deflate")
    except Exception:
        shutil.copy(tif, cog_dir / tif.name)
print("COGs:", [p.name for p in cog_dir.glob("*.tif")])

# %% [markdown]
# ## 4. Interactive web map

# %%
# Cell 10 - Reproject a layer to lat/lon and colour it so Leaflet can show it
def to_overlay(arr, cmap_name, vmin, vmax):
    left, bottom, right, top = GRID["xmin"], GRID["ymin"], GRID["xmax"], GRID["ymax"]
    dst_tf, dw, dh = calculate_default_transform(GRID["crs"], "EPSG:4326", W, H, left, bottom, right, top)
    dst = np.full((dh, dw), np.nan, "float32")
    reproject(arr.astype("float32"), dst, src_transform=TRANSFORM, src_crs=GRID["crs"], dst_transform=dst_tf,
              dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    w_, s_, e_, n_ = rasterio.transform.array_bounds(dh, dw, dst_tf)
    rgba = plt.get_cmap(cmap_name)(np.clip((dst - vmin) / (vmax - vmin), 0, 1))
    rgba[~np.isfinite(dst)] = 0
    return rgba, [[s_, w_], [n_, e_]]

print("helper ready")

# %%
# Cell 11 - Build and save the folium map
west, south, east, north = CFG["bbox"]
fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=10, tiles="OpenStreetMap")
folium.GeoJson(
    CFG["aoi"]["geometry"], name=CFG["aoi"]["name"],
    style_function=lambda _: {"color": "#1769aa", "weight": 2, "fillOpacity": 0.02},
).add_to(fmap)
for name, arr, cmap_name, lo, hi, show_now in [("InSAR velocity (mm/yr)", vel_obs, "RdBu", -30, 30, True),
                                                ("RF severe-subsidence probability", p_severe, "OrRd", 0, 1, False),
                                                ("Mean stress score", stress_mean, "RdYlBu_r", -0.6, 0.6, False),
                                                ("Hotspot classes", hot, "viridis", 0, 3, False)]:
    rgba, bounds = to_overlay(arr, cmap_name, lo, hi)
    folium.raster_layers.ImageOverlay(image=rgba, bounds=bounds, opacity=0.75, name=name, show=show_now).add_to(fmap)
for name, (lon, lat) in CFG["points"].items():
    folium.Marker([lat, lon], tooltip=name).add_to(fmap)
folium.LayerControl().add_to(fmap)
if MODE == "demo":
    fmap.get_root().html.add_child(folium.Element('<div style="position:fixed;top:10px;left:60px;z-index:9999;background:#fff;color:#c00;padding:4px 8px;font-weight:bold">SYNTHETIC DEMO DATA</div>'))
fmap.save(MAPS / "naivasha_interactive.html")
print("saved", MAPS / "naivasha_interactive.html")

# %% [markdown]
# ## 5. Auto-generated results summary

# %%
# Cell 12 - Write RESULTS_SUMMARY.md using the real numbers from notebooks 06 and 07
f2 = lambda v: f"{v:.2f}"
sub_cv = [r for r in m06["subsidence_regression_cv"] if r["cv"] == "spatial blocks"]
best_sub = min(sub_cv, key=lambda r: r["rmse"])
inflated = [r for r in m06["subsidence_regression_cv"] if "inflated" in r["cv"]][0]
fc = pd.DataFrame(m06["stress_forecast"])
lines = []
if MODE == "demo":
    lines += ["> **SYNTHETIC DEMO RESULTS.** These numbers come from invented data and only prove the pipeline runs. Do not cite them.", ""]
lines += [f"# Results summary ({MODE} mode)", "", f"Study period {YEARS[0]}-{YEARS[-1]}, grid {W} x {H} cells at {GRID['res']} m.", "",
          "## Subsidence",
          f"- InSAR stable-area null test: mean {f2(m07['insar_null_test']['mean_mm_yr'])} mm/yr, robust SD {f2(m07['insar_null_test']['robust_sd_mm_yr'])} mm/yr "
          f"(n = {m07['insar_null_test']['n_stable']}). Significant-subsidence threshold: {f2(m07['insar_null_test']['subsidence_threshold_mm_yr'])} mm/yr.",
          f"- Random Forest velocity (spatial-block CV, feature set `{best_sub['feature_set']}`): R2 {f2(best_sub['r2'])}, RMSE {f2(best_sub['rmse'])} mm/yr, MAE {f2(best_sub['mae'])} mm/yr.",
          f"- For comparison, ordinary random k-fold gives R2 {f2(inflated['r2'])} - the gap is the inflation caused by spatial autocorrelation.",
          f"- Severity classes (spatial CV): macro-F1 {f2(m06['subsidence_classification_cv']['f1_macro'])}, kappa {f2(m06['subsidence_classification_cv']['kappa'])}.", ""]
if "gnss_check" in m07:
    g = m07["gnss_check"]
    lines += [f"- GNSS/levelling check (n = {g['n']}): bias {f2(g['bias_mm_yr'])} mm/yr, RMSE {f2(g['rmse_mm_yr'])} mm/yr, r = {f2(g['r'])}.", ""]
lines += ["## Agricultural stress forecast", f"Forecast horizon {CFG['forecast_horizon']} months; hold-out from {CFG['holdout_start_year']}.", "",
          "| Model | Test | RMSE | R2 | Skill vs climatology |", "|---|---|---|---|---|"]
for _, r in fc.iterrows():
    lines.append(f"| {r['model']} | {r['test']} | {f2(r['rmse'])} | {f2(r['r2'])} | {f2(r['skill_vs_climatology'])} |")
lines += ["", f"- Climate alone explains about {100 * m07['climate_share_of_stress_variance']:.0f}% of month-to-month stress variance (out-of-fold R2).", "",
          "## Coupling", "| Velocity vs | n | Spearman r | 95% CI (block bootstrap) |", "|---|---|---|---|"]
for r in m07["spatial_coupling"]:
    lines.append(f"| {r['velocity vs']} | {r['n']} | {f2(r['spearman_r'])} | {f2(r['ci95_low'])} to {f2(r['ci95_high'])} |")
h = m07["hotspots"]
lines += ["", f"- Hotspot area (km2): " + ", ".join(f"{k} {v:.0f}" for k, v in h["area_km2"].items()) + f". Co-occurrence is {h['overlap_ratio_vs_chance']:.1f}x what independence would give.", ]
if "lag_correlation" in m07:
    l = m07["lag_correlation"]
    lines.append(f"- Strongest lag correlation at {l['best_lag_months']} months (r = {f2(l['r'])}, adjusted p = {l['p_adj']:.3f}); positive lag means the sinking rate changes after the stress anomaly.")
lines += ["", "## Files", "- `outputs/maps/` rasters, COGs, time-lapse GIF, interactive HTML map", "- `outputs/figures/` PNG figures", "- `outputs/tables/` CSV/JSON metrics", "- `outputs/models/` trained Random Forests (joblib)", ""]
(OUT / "RESULTS_SUMMARY.md").write_text("\n".join(lines))
print("\n".join(lines[:30]))

# %%
# Cell 13 - List everything produced
for p in sorted(OUT.rglob("*")):
    if p.is_file() and "cog" not in p.parts:
        print(f"{p.relative_to(OUT)!s:60s} {p.stat().st_size / 1e3:9.1f} kB")
