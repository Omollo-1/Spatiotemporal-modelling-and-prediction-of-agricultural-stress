# %% [markdown]
# # 07 - Validation, coupling and hotspots
# This notebook answers the scientific questions and checks how much to trust the answers.
#
# 1. **InSAR quality** - how noisy is the velocity map? (stable-area null test, optional GNSS/levelling comparison)
# 2. **Is subsidence spatially coupled with crop stress?** - Spearman correlation with a *block bootstrap* confidence interval
#    (so spatial autocorrelation does not make the result look more certain than it is).
# 3. **Is the coupling still there once climate is removed?** - using the non-climatic stress from notebook 06.
# 4. **Hotspots** - where are subsidence and stress both high?
# 5. **Timing** - does stress lead or lag the rate of ground sinking? (lag correlation with an effective-sample-size correction)
#
# > Correlation is not causation. Coupling here means "occurs together in space/time", which is consistent with, but does not prove,
# > a groundwater-abstraction mechanism. Tectonic and volcanic signals can produce deformation with no link to farming.

# @@boot_env@@

# @@boot_cfg@@

# @@io_helpers@@

# %%
# Cell 4 - Imports, settings and tables
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
ensure("scipy", "pyarrow", "matplotlib", "pyproj")
import matplotlib.pyplot as plt
from scipy import stats
from pyproj import Transformer

DER, TAB, FIG = DATA / "derived", OUT / "tables", OUT / "figures"
rng = np.random.default_rng(CFG["seed"])
T = len(MONTHS)
static = pd.read_parquet(DER / "static_pixels.parquet")
meta = json.loads((DER / "feature_meta.json").read_text())
m06 = json.loads((TAB / "06_metrics.json").read_text())
stress_cube = np.load(DER / "stress_score.npy")                    # [T, H, W]
RESULTS = {"mode": MODE}
print(f"{len(static):,} land pixels | stress cube {stress_cube.shape}")

# %% [markdown]
# ## 1. InSAR quality: stable-area null test

# %%
# Cell 5 - Pick "stable" pixels (flat, far from the lake, geothermal field and volcano, not cropland/built-up)
def col(name, default=np.inf):
    return static[name] if name in static.columns else pd.Series(default, index=static.index)


stable = static[(static["slope"] < 2) & (col("dist_lake_km") > 8) & (col("dist_olkaria_geothermal_km") > 12) &
                (col("dist_mt_longonot_km") > 10) & (static["crop_frac"] < 0.2) & (static["built_frac"] < 0.2) &
                np.isfinite(static["vel"])]
if len(stable) < 30:
    print("Few stable pixels found - using all valid pixels for a robust (median/MAD) estimate instead.")
    stable = static[np.isfinite(static["vel"])]
mu = float(np.median(stable["vel"]))
sigma = float(1.4826 * np.median(np.abs(stable["vel"] - mu)))      # robust standard deviation
V_THRESH = min(CFG["subsidence_class_breaks"][0], -2 * sigma)       # "significant subsidence" threshold (mm/yr)
RESULTS["insar_null_test"] = dict(n_stable=int(len(stable)), mean_mm_yr=mu, robust_sd_mm_yr=sigma, subsidence_threshold_mm_yr=V_THRESH)
print(f"stable pixels: {len(stable):,} | mean {mu:.2f} mm/yr | robust SD {sigma:.2f} mm/yr | threshold {V_THRESH:.1f} mm/yr")
print("median velocity uncertainty reported by the processor: %.2f mm/yr" % np.nanmedian(static["vel_std"]))

# %% [markdown]
# ## 2. Optional: compare with GNSS or levelling points
# Put a CSV with columns `lon, lat, vel_up_mm_yr` in `data/ancillary/gnss.csv`. Skipped if the file is missing.

# %%
# Cell 6 - GNSS / levelling check
gnss_file = DATA / "ancillary" / "gnss.csv"
if gnss_file.exists():
    g = pd.read_csv(gnss_file)
    to_utm = Transformer.from_crs("EPSG:4326", GRID["crs"], always_xy=True)
    gx, gy = to_utm.transform(g["lon"].to_numpy(), g["lat"].to_numpy())
    vel_map = read_stack(DATA / "insar" / "velocity_vertical_mm_yr.tif")[0][0]
    vals = []
    for x_, y_ in zip(gx, gy):
        c, r = int((x_ - GRID["xmin"]) // GRID["res"]), int((GRID["ymax"] - y_) // GRID["res"])
        win = vel_map[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
        vals.append(np.nanmedian(win) if np.isfinite(win).any() else np.nan)
    g["insar"] = vals
    ok = g.dropna(subset=["insar"])
    if len(ok) >= 3:
        RESULTS["gnss_check"] = dict(n=int(len(ok)), bias_mm_yr=float((ok["insar"] - ok["vel_up_mm_yr"]).mean()),
                                    rmse_mm_yr=float(np.sqrt(((ok["insar"] - ok["vel_up_mm_yr"]) ** 2).mean())),
                                    r=float(np.corrcoef(ok["insar"], ok["vel_up_mm_yr"])[0, 1]))
        print(RESULTS["gnss_check"])
    else:
        print("Fewer than 3 GNSS points fall on valid InSAR pixels - check coordinates.")
else:
    print("No gnss.csv found - skipping. (Validation against ground truth is the strongest check you can add.)")

# %% [markdown]
# ## 3. Spatial coupling between subsidence and stress

# %%
# Cell 7 - Join pixel table with the non-climatic stress from notebook 06
resid = pd.read_csv(TAB / "06_pixel_nonclimatic_stress.csv")
px = static.merge(resid, on="pix", how="left")
ag_px = px[px["is_ag"] & np.isfinite(px["vel"]) & np.isfinite(px["stress_mean"])].reset_index(drop=True)
print(f"agricultural pixels with InSAR + stress: {len(ag_px):,} | with non-climatic stress: {int(ag_px['nonclimatic_mean'].notna().sum()):,}")

# %%
# Cell 8 - Spearman correlation with a spatial-block bootstrap confidence interval
def block_boot_ci(a, b, blocks, n_boot=300):
    ub = np.unique(blocks)
    idx_by = {k: np.flatnonzero(blocks == k) for k in ub}
    rs = []
    for _ in range(n_boot):
        pick = rng.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([idx_by[k] for k in pick])
        rs.append(stats.spearmanr(a[idx], b[idx])[0])
    return np.percentile(rs, [2.5, 97.5])


rows = []
# NOTE: anomalies are measured against the same period's climatology, so the MEAN stress is ~0 for every pixel by construction.
# A chronic worsening shows up in the TREND metrics, which are therefore the primary coupling tests.
for label, other in [("stress trend (z per year)  [primary]", "stress_trend"),
                     ("non-climatic stress trend (z per year)  [primary]", "nonclimatic_trend"),
                     ("share of months in high stress", "stress_frac_high"),
                     ("mean stress score", "stress_mean")]:
    d = ag_px[["vel", other, "block"]].dropna()
    if len(d) < 30:
        continue
    r = stats.spearmanr(d["vel"], d[other])[0]
    lo, hi = block_boot_ci(d["vel"].to_numpy(), d[other].to_numpy(), d["block"].to_numpy())
    rows.append({"velocity vs": label, "n": len(d), "spearman_r": r, "ci95_low": lo, "ci95_high": hi,
                 "ci_excludes_0": bool(lo > 0 or hi < 0)})
coupling = pd.DataFrame(rows)
coupling.to_csv(TAB / "07_spatial_coupling.csv", index=False)
RESULTS["spatial_coupling"] = rows
print("Negative r = faster sinking goes with more stress.")
coupling.round(3)

# %% [markdown]
# ## 4. Hotspots: subsiding AND stressed

# %%
# Cell 9 - Classify pixels: 0 neither | 1 subsiding only | 2 stressed only | 3 both
STRESS_METRIC = "stress_trend"   # persistent worsening of vegetation condition (alternatives: "stress_frac_high", "ndvi_trend" x -1)
STRESS_HOT_PCTL = 75             # "stressed" = top quartile of that metric among agricultural pixels
thr_stress = np.nanpercentile(static.loc[static["is_ag"], STRESS_METRIC], STRESS_HOT_PCTL)
sub = (static["vel"] < V_THRESH).to_numpy()
strs = (static["is_ag"] & (static[STRESS_METRIC] >= thr_stress)).to_numpy()
hot = sub.astype(int) + 2 * strs.astype(int)
hot_map = np.full(H * W, np.nan, dtype="float32")
hot_map[static["pix"].to_numpy()] = hot
write_stack(OUT / "maps" / "hotspot_classes.tif", hot_map.reshape(H, W))
cell_km2 = (GRID["res"] / 1000) ** 2
area = {n: float((hot == k).sum() * cell_km2) for k, n in enumerate(["neither", "subsiding_only", "stressed_only", "both"])}
valid = np.isfinite(static["vel"]).to_numpy() & static["is_ag"].to_numpy()
p_sub, p_str = sub[valid].mean(), strs[valid].mean()
overlap = (sub & strs)[valid].mean() / max(p_sub * p_str, 1e-9)      # >1 means they co-occur more than chance
RESULTS["hotspots"] = dict(area_km2=area, stress_metric=STRESS_METRIC, stress_threshold=float(thr_stress), overlap_ratio_vs_chance=float(overlap))
print("area (km2):", {k: round(v, 1) for k, v in area.items()})
print(f"co-occurrence vs chance (agricultural pixels with InSAR): {overlap:.2f}x  (1 = independent)")

# %% [markdown]
# ## 5. Timing: does stress lead or lag the sinking?

# %%
# Cell 10 - Monthly series: displacement rate in subsiding pixels vs stress anomaly (climatic and non-climatic)
disp_file = DATA / "insar" / "disp_vertical_monthly_mm.tif"
series = None
if disp_file.exists():
    disp = read_stack(disp_file)[0]                                           # [T, H, W] mm, relative to first epoch
    sub_map = np.zeros(H * W, bool); sub_map[static["pix"].to_numpy()] = sub
    ag_map = np.zeros(H * W, bool); ag_map[static["pix"].to_numpy()] = static["is_ag"].to_numpy()
    sub_map, ag_map = sub_map.reshape(H, W), ag_map.reshape(H, W)
    d_sub = np.nanmean(disp[:, sub_map], axis=1)                              # mean displacement in subsiding pixels
    series = pd.DataFrame({"disp_mm": d_sub, "rate_mm_month": np.r_[np.nan, np.diff(d_sub)],
                           "stress_subsiding": np.nanmean(stress_cube[:, sub_map & ag_map], axis=1),
                           "stress_reference": np.nanmean(stress_cube[:, ag_map & ~sub_map], axis=1)}, index=MONTHS)
    rs = pd.read_parquet(DER / "panel_climate_resid.parquet")
    hot_pix = set(static.loc[sub, "pix"])
    rs["in_sub"] = rs["pix"].isin(hot_pix)
    nc = rs[rs["in_sub"]].groupby("origin")["nonclimatic"].mean()
    series["nonclimatic_subsiding"] = pd.Series(nc.values, index=[MONTHS[i] for i in nc.index]).reindex(MONTHS)
    series.to_csv(TAB / "07_monthly_series.csv")
    print(series.dropna().tail(3).round(2))
else:
    print("No monthly displacement file - skipping timing analysis (needs 04 to export disp_vertical_monthly_mm.tif).")

# %%
# Cell 11 - Lag correlation corr(stress_t, rate_{t+lag}) with an effective sample size that accounts for autocorrelation
def lag_corr(x, y, lags=range(-6, 7)):
    out = []
    for k in lags:
        d = pd.concat([x, y.shift(-k)], axis=1).dropna()
        if len(d) < 24:
            out.append((k, np.nan, np.nan, len(d))); continue
        r = np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]
        a1 = d.iloc[:, 0].autocorr(1); a2 = d.iloc[:, 1].autocorr(1)
        n_eff = max(len(d) * (1 - a1 * a2) / (1 + a1 * a2), 3)               # Bretherton et al. (1999)
        t = r * np.sqrt((n_eff - 2) / max(1 - r ** 2, 1e-9))
        out.append((k, r, 2 * (1 - stats.t.cdf(abs(t), n_eff - 2)), n_eff))
    return pd.DataFrame(out, columns=["lag_months", "r", "p_adj", "n_eff"])


lagtab = None
if series is not None:
    x = series["nonclimatic_subsiding"] if series["nonclimatic_subsiding"].notna().sum() > 24 else series["stress_subsiding"]
    lagtab = lag_corr(x, series["rate_mm_month"])
    lagtab.to_csv(TAB / "07_lag_correlation.csv", index=False)
    best = lagtab.loc[lagtab["r"].abs().idxmax()]
    RESULTS["lag_correlation"] = dict(best_lag_months=int(best["lag_months"]), r=float(best["r"]), p_adj=float(best["p_adj"]),
                                      note="positive lag = sinking rate changes AFTER the stress anomaly")
    print(RESULTS["lag_correlation"])

# %% [markdown]
# ## 6. How much of the stress is climate?

# %%
# Cell 12 - Share of stress variance explained by climate alone (out-of-fold R2 from notebook 06)
r2c = m06["climate_only_model"]["r2"]
RESULTS["climate_share_of_stress_variance"] = float(r2c)
print(f"Climate explains about {100 * r2c:.0f}% of month-to-month stress variance (spatial out-of-fold R2); the rest is non-climatic or noise.")

# %%
# Cell 13 - Summary figure
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#eeeeee", "#4575b4", "#fdae61", "#d73027"])
im = ax[0, 0].imshow(np.where(np.isfinite(hot_map.reshape(H, W)), hot_map.reshape(H, W), np.nan), cmap=cmap, vmin=-0.5, vmax=3.5)
ax[0, 0].set_title("Hotspots"); ax[0, 0].axis("off")
cb = plt.colorbar(im, ax=ax[0, 0], ticks=[0, 1, 2, 3], shrink=0.7); cb.ax.set_yticklabels(["neither", "subsiding", "stressed", "both"])
ax[0, 1].hexbin(ag_px["vel"], ag_px[STRESS_METRIC], gridsize=40, cmap="viridis", bins="log", mincnt=1)
ax[0, 1].axvline(V_THRESH, color="r", ls="--", lw=1); ax[0, 1].set_xlabel("InSAR velocity (mm/yr)"); ax[0, 1].set_ylabel(STRESS_METRIC)
ax[0, 1].set_title("Agricultural pixels: velocity vs stress")
if lagtab is not None:
    ax[1, 0].bar(lagtab["lag_months"], lagtab["r"], color=np.where(lagtab["p_adj"] < 0.05, "tab:red", "tab:gray"))
    ax[1, 0].set_xlabel("Lag (months); positive = stress leads"); ax[1, 0].set_ylabel("Correlation"); ax[1, 0].set_title("Stress vs sinking rate (red: p<0.05)")
    ax[1, 1].plot(series["disp_mm"].to_numpy(), label="mean displacement (mm)")
    ax2 = ax[1, 1].twinx(); ax2.plot(series["stress_subsiding"].to_numpy(), "r", alpha=0.6, label="stress in subsiding pixels")
    ax[1, 1].set_xticks(range(0, T, 12)); ax[1, 1].set_xticklabels(YEARS, rotation=45); ax[1, 1].set_title("Subsiding pixels through time")
else:
    ax[1, 0].axis("off"); ax[1, 1].axis("off")
fig.savefig(FIG / "07_coupling_summary.png", dpi=140, bbox_inches="tight")
plt.show()

# %%
# Cell 14 - Save results
(TAB / "07_results.json").write_text(json.dumps(RESULTS, indent=2, default=float))
print("saved", TAB / "07_results.json")

# %% [markdown]
# **Next:** `08_maps_and_report`.
