# %% [markdown]
# # 06 - Random Forest models
# Two model families, both validated with **spatial-block cross-validation** (neighbouring pixels never sit in both the
# training and the test fold, because they are near-copies of each other and would inflate accuracy).
#
# **A. Land-subsidence model** - target: InSAR vertical velocity (mm/yr) and a 3-class severity (stable / moderate / severe).
# Two feature sets are compared: *drivers only* (terrain, soil, land cover, distances) and *drivers + vegetation/climate summaries*.
#
# **B. Agricultural-stress forecast** - target: stress score `H` months ahead. Trained on targets before the hold-out year and
# tested on later years (a true forecast). Compared against two baselines (climatology = 0 and persistence) and a gradient-boosting model.
# A *strict* test also hides whole spatial blocks, so the model is judged on places and times it has never seen.
#
# **C. Climate-only model** - how much stress does climate alone explain? Its out-of-fold residual is the *non-climatic stress*
# (a proxy for abstraction / management effects) used in notebook 07 for the coupling analysis.

# @@boot_env@@

# @@boot_cfg@@

# @@io_helpers@@

# %%
# Cell 4 - Imports and settings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
ensure("scikit-learn", "shap", "joblib", "pyarrow", "matplotlib")
import joblib, shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, f1_score, cohen_kappa_score,
                             confusion_matrix)
from sklearn.inspection import permutation_importance

SEED = CFG["seed"]
TREES = CFG["rf_trees"]
DER, TAB, FIG, MOD = DATA / "derived", OUT / "tables", OUT / "figures", OUT / "models"
rng = np.random.default_rng(SEED)
METRICS = {}                       # everything we report is collected here and saved at the end
print("trees:", TREES, "| seed:", SEED)

# %%
# Cell 5 - Load the tables from notebook 05
static = pd.read_parquet(DER / "static_pixels.parquet")
panel = pd.read_parquet(DER / "panel_stress.parquet")
meta = json.loads((DER / "feature_meta.json").read_text())
DRIVERS, VEG, DYN = meta["driver_features"], meta["veg_features"], meta["dynamic_features"]
print(f"static pixels {len(static):,} | panel rows {len(panel):,}")
print("driver features:", DRIVERS)

# %%
# Cell 6 - Helper functions: metrics, model factories, spatial cross-validation
def reg_metrics(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return dict(rmse=float(np.sqrt(mean_squared_error(y[ok], p[ok]))), mae=float(mean_absolute_error(y[ok], p[ok])),
                r2=float(r2_score(y[ok], p[ok])), n=int(ok.sum()))


def clf_metrics(y, p):
    return dict(f1_macro=float(f1_score(y, p, average="macro")), kappa=float(cohen_kappa_score(y, p)),
                accuracy=float((y == p).mean()), confusion=confusion_matrix(y, p, labels=[0, 1, 2]).tolist())


def make_rf(kind="reg", max_samples=None, leaf=5, depth=None, mf=0.5):
    cls = RandomForestRegressor if kind == "reg" else RandomForestClassifier
    kw = dict(n_estimators=TREES, min_samples_leaf=leaf, max_depth=depth, max_features=mf, n_jobs=-1, random_state=SEED,
              max_samples=max_samples)
    if kind == "clf":
        kw["class_weight"] = "balanced_subsample"
    return cls(**kw)


def cv_predict(make_model, X, y, groups, spatial=True, n_splits=5):
    """Out-of-fold predictions. spatial=True keeps each block entirely inside one fold."""
    oof = np.full(len(y), np.nan)
    splitter = GroupKFold(n_splits=n_splits) if spatial else KFold(n_splits, shuffle=True, random_state=SEED)
    for tr, te in splitter.split(X, y, groups):
        med = X.iloc[tr].median()
        model = make_model().fit(X.iloc[tr].fillna(med), y[tr])
        oof[te] = model.predict(X.iloc[te].fillna(med))
    return oof

print("helpers ready")

# %% [markdown]
# ## A. Land-subsidence model

# %%
# Cell 7 - Training pixels: land pixels with a valid InSAR velocity (optionally subsampled for speed)
FEATURE_SETS: dict[str, list[str]] = {
    "A_drivers_only": DRIVERS,
    "B_drivers_plus_vegetation": DRIVERS + VEG,
}
train = static[np.isfinite(static["vel"])].copy()
if len(train) > 60000:
    train = train.sample(60000, random_state=SEED)
y_vel = train["vel"].to_numpy()
groups = train["block"].to_numpy()
print(f"training pixels: {len(train):,} | velocity mean {y_vel.mean():.2f}, min {y_vel.min():.1f}, max {y_vel.max():.1f} mm/yr")

# %%
# Cell 8 - Spatial-block CV for both feature sets, plus ordinary random CV to show how much it inflates accuracy
rows, oof = [], {}
for name, feats in FEATURE_SETS.items():
    X = train[feats]
    oof[name] = cv_predict(lambda: make_rf("reg"), X, y_vel, groups, spatial=True)
    rows.append({"feature_set": name, "cv": "spatial blocks", **reg_metrics(y_vel, oof[name])})
rand = cv_predict(lambda: make_rf("reg"), train[FEATURE_SETS["B_drivers_plus_vegetation"]], y_vel, groups, spatial=False)
rows.append({"feature_set": "B_drivers_plus_vegetation", "cv": "random k-fold (inflated)", **reg_metrics(y_vel, rand)})
subs_cv = pd.DataFrame(rows)
subs_cv.to_csv(TAB / "06_subsidence_cv.csv", index=False)
METRICS["subsidence_regression_cv"] = rows
subs_cv.round(3)

# %%
# Cell 9 - Choose the primary feature set (lowest spatial-CV RMSE) and measure permutation importance on unseen blocks
spatial_rows = subs_cv[subs_cv["cv"] == "spatial blocks"].set_index("feature_set")
PRIMARY = str(spatial_rows["rmse"].idxmin())
FEATS = FEATURE_SETS[PRIMARY]
test_blocks = rng.choice(np.unique(groups), size=max(1, int(0.2 * np.unique(groups).size)), replace=False)
te = np.isin(groups, test_blocks)
med = train.loc[~te, FEATS].median()
m = make_rf("reg").fit(train.loc[~te, FEATS].fillna(med), y_vel[~te])
sub = train.loc[te, FEATS].fillna(med)
sub_idx = rng.choice(len(sub), size=min(3000, len(sub)), replace=False)
pi = permutation_importance(m, sub.iloc[sub_idx], y_vel[te][sub_idx], n_repeats=3, random_state=SEED, n_jobs=-1, scoring="r2")
imp = pd.DataFrame({"feature": FEATS, "perm_importance_r2_drop": np.asarray(pi["importances_mean"])}).sort_values(
    "perm_importance_r2_drop", ascending=False)
imp.to_csv(TAB / "06_subsidence_permutation_importance.csv", index=False)
METRICS["subsidence_primary_set"] = PRIMARY
print("primary feature set:", PRIMARY)
imp.head(10).round(4)

# %%
# Cell 10 - Final subsidence regression model + SHAP (which drivers push velocity up or down)
med_all = train[FEATS].median()
rf_vel = make_rf("reg").fit(train[FEATS].fillna(med_all), y_vel)
shap_rows = train[FEATS].fillna(med_all).sample(min(1500, len(train)), random_state=SEED)
sv = shap.TreeExplainer(rf_vel).shap_values(shap_rows, check_additivity=False)
plt.figure()
shap.summary_plot(sv, shap_rows, show=False, max_display=15)
plt.title("SHAP: drivers of subsidence velocity (negative = sinking)")
plt.savefig(FIG / "06_shap_subsidence.png", dpi=150, bbox_inches="tight")
plt.show()
pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(
    TAB / "06_subsidence_shap.csv", index=False)

# %%
# Cell 11 - Subsidence SEVERITY classifier: stable / moderate / severe (spatial-block CV)
b_mod, b_sev = CFG["subsidence_class_breaks"]               # e.g. -5 and -15 mm/yr
to_class = lambda v: np.select([v > b_mod, v > b_sev], [0, 1], default=2)
y_cls = to_class(y_vel)
print("class counts (0 stable, 1 moderate, 2 severe):", np.bincount(y_cls, minlength=3))
oof_cls = np.full(len(y_cls), -1)
for tr, te_ in GroupKFold(n_splits=5).split(train, y_cls, groups):
    med = train.iloc[tr][FEATS].median()
    clf = make_rf("clf").fit(train.iloc[tr][FEATS].fillna(med), y_cls[tr])
    oof_cls[te_] = clf.predict(train.iloc[te_][FEATS].fillna(med))
METRICS["subsidence_classification_cv"] = clf_metrics(y_cls, oof_cls)
print({k: round(v, 3) for k, v in METRICS["subsidence_classification_cv"].items() if k != "confusion"})
print("confusion matrix (rows = observed, cols = predicted):")
print(np.array(METRICS["subsidence_classification_cv"]["confusion"]))

# %%
# Cell 12 - Predict EVERY land pixel (including areas where InSAR decorrelates, e.g. greenhouses) and save rasters
clf_final = make_rf("clf").fit(train[FEATS].fillna(med_all), y_cls)
Xall = static[FEATS].fillna(med_all)
pred_vel = rf_vel.predict(Xall).astype("float32")
proba = clf_final.predict_proba(Xall)
p_severe = proba[:, list(clf_final.classes_).index(2)].astype("float32") if 2 in clf_final.classes_ else np.zeros(len(Xall), "float32")
pred_cls = clf_final.predict(Xall).astype("float32")


def vec_to_map(v):
    out = np.full(H * W, np.nan, dtype="float32")
    out[static["pix"].to_numpy()] = v
    return out.reshape(H, W)


write_stack(OUT / "maps" / "pred_velocity_mm_yr.tif", vec_to_map(pred_vel))
write_stack(OUT / "maps" / "pred_subsidence_class.tif", vec_to_map(pred_cls))
write_stack(OUT / "maps" / "pred_p_severe.tif", vec_to_map(p_severe))
oof_full = np.full(len(static), np.nan, dtype="float32")
oof_full[static.index.get_indexer(train.index)] = oof[PRIMARY]
write_stack(OUT / "maps" / "oof_velocity_mm_yr.tif", vec_to_map(oof_full))
print("saved prediction rasters in", OUT / "maps")

# %%
# Cell 13 - Observed vs predicted (out-of-fold, spatial CV)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
lim = [min(y_vel.min(), -5), max(y_vel.max(), 5)]
ax[0].hexbin(y_vel, oof[PRIMARY], gridsize=40, cmap="viridis", mincnt=1, bins="log")
ax[0].plot(lim, lim, "r--", lw=1); ax[0].set_xlabel("InSAR velocity (mm/yr)"); ax[0].set_ylabel("RF prediction (mm/yr)")
r = spatial_rows.loc[PRIMARY]
ax[0].set_title(f"Spatial-CV: R2 {r['r2']:.2f}, RMSE {r['rmse']:.2f} mm/yr")
cm = np.array(METRICS["subsidence_classification_cv"]["confusion"])
ax[1].imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax[1].text(j, i, cm[i, j], ha="center", va="center")
ax[1].set_xticks(range(3)); ax[1].set_yticks(range(3)); ax[1].set_xticklabels(["stable", "moderate", "severe"]); ax[1].set_yticklabels(["stable", "moderate", "severe"])
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Observed"); ax[1].set_title("Severity confusion matrix")
fig.savefig(FIG / "06_subsidence_validation.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## B. Agricultural-stress forecast

# %%
# Cell 14 - Time split: train on targets BEFORE the hold-out year, test on the hold-out years
HOLD = CFG["holdout_start_year"]
tr_mask = panel["target_year"] < HOLD
te_mask = ~tr_mask
SETS_F = {"F1_dynamics_only": DYN,
          "F2_plus_drivers": DYN + DRIVERS,
          "F3_plus_subsidence_velocity": DYN + DRIVERS + ["vel"]}
y_all = panel["target"].to_numpy()
print(f"train rows {tr_mask.sum():,} (targets < {HOLD}) | test rows {te_mask.sum():,} (targets >= {HOLD})")

# %%
# Cell 15 - Baselines: climatology (anomaly = 0) and persistence (stress now = stress in H months)
rows_f = []
y_te = y_all[te_mask]
rows_f.append({"model": "Baseline: climatology (0)", "test": "future years", **reg_metrics(y_te, np.zeros_like(y_te))})
rows_f.append({"model": "Baseline: persistence", "test": "future years", **reg_metrics(y_te, panel.loc[te_mask, "stress"].to_numpy())})
pd.DataFrame(rows_f).round(3)

# %%
# Cell 16 - Random Forest and gradient boosting on each feature set (future-years test)
# Only ~8 training years x 12 months = ~100 INDEPENDENT regional time points, however many pixels there are. An unconstrained
# forest memorises those months (negative skill on new years), so the forecast forest is deliberately shallow and coarse.
MS = 0.25 if len(panel) > 150000 else None          # 25 % of rows per tree on big panels (faster)
FC_LEAF = max(20, int(0.005 * tr_mask.sum()))
print("forecast forest settings:", {"max_samples": MS, "leaf": FC_LEAF, "depth": 8, "max_features": 0.3})
fitted, preds = {}, {}
for name, feats in SETS_F.items():
    Xtr, Xte = panel.loc[tr_mask, feats], panel.loc[te_mask, feats]
    med = Xtr.median()
    rf = make_rf("reg", max_samples=MS, leaf=FC_LEAF, depth=8, mf=0.3).fit(Xtr.fillna(med), y_all[tr_mask])
    fitted[name], preds[name] = (rf, med), rf.predict(Xte.fillna(med))
    rows_f.append({"model": f"RF {name}", "test": "future years", **reg_metrics(y_te, preds[name])})
hgb = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=4, min_samples_leaf=FC_LEAF, random_state=SEED)
Xtr, Xte = panel.loc[tr_mask, SETS_F["F2_plus_drivers"]], panel.loc[te_mask, SETS_F["F2_plus_drivers"]]
hgb.fit(Xtr, y_all[tr_mask])
rows_f.append({"model": "HistGradientBoosting F2_plus_drivers", "test": "future years", **reg_metrics(y_te, hgb.predict(Xte))})
forecast_tab = pd.DataFrame(rows_f)
forecast_tab.round(3)

# %%
# Cell 17 - STRICT test: future years AND spatial blocks the model has never seen
all_blocks = panel["block"].unique()
held = rng.choice(all_blocks, size=max(1, int(0.2 * all_blocks.size)), replace=False)
in_held = panel["block"].isin(held)
strict_tr, strict_te = tr_mask & ~in_held, te_mask & in_held
for name, feats in SETS_F.items():
    med = panel.loc[strict_tr, feats].median()
    rf = make_rf("reg", max_samples=MS, leaf=FC_LEAF, depth=8, mf=0.3).fit(
        panel.loc[strict_tr, feats].fillna(med), y_all[strict_tr])
    p = rf.predict(panel.loc[strict_te, feats].fillna(med))
    rows_f.append({"model": f"RF {name}", "test": "future years + unseen blocks", **reg_metrics(y_all[strict_te], p)})
rows_f.append({"model": "Baseline: persistence", "test": "future years + unseen blocks",
               **reg_metrics(y_all[strict_te], panel.loc[strict_te, "stress"].to_numpy())})
rows_f.append({"model": "Baseline: climatology (0)", "test": "future years + unseen blocks",
               **reg_metrics(y_all[strict_te], np.zeros(strict_te.sum()))})
forecast_tab = pd.DataFrame(rows_f)
# Skill score vs climatology: 1 = perfect, 0 = no better than "a normal month", < 0 = worse than climatology
clim_rmse = forecast_tab[forecast_tab["model"].str.contains("climatology")].set_index("test")["rmse"]
forecast_tab["skill_vs_climatology"] = 1 - (forecast_tab["rmse"] / forecast_tab["test"].map(clim_rmse)) ** 2
rows_f = forecast_tab.to_dict("records")
forecast_tab.to_csv(TAB / "06_stress_forecast_metrics.csv", index=False)
METRICS["stress_forecast"] = rows_f
forecast_tab.round(3)

# %%
# Cell 18 - Spatial-block CV on the TRAINING period (F2) - how stable is the model across space?
Xtr2 = panel.loc[tr_mask, SETS_F["F2_plus_drivers"]]
oof_f = cv_predict(lambda: make_rf("reg", max_samples=MS, leaf=FC_LEAF, depth=8, mf=0.3),
                   Xtr2, y_all[tr_mask], panel.loc[tr_mask, "block"].to_numpy(), spatial=True)
METRICS["stress_forecast_spatial_cv_train_period"] = reg_metrics(y_all[tr_mask], oof_f)
print({k: round(v, 3) for k, v in METRICS["stress_forecast_spatial_cv_train_period"].items()})

# %%
# Cell 19 - Classify forecasts into low / moderate / high stress and score them
b_lo, b_hi = CFG["stress_class_breaks"]
sclass = lambda s: np.digitize(s, [b_lo, b_hi])
best = "F2_plus_drivers"
cls_true = sclass(y_te)
cls_rf = sclass(preds[best])
cls_persist = sclass(panel.loc[te_mask, "stress"].fillna(0).to_numpy())
METRICS["stress_class_rf"] = clf_metrics(cls_true, cls_rf)
METRICS["stress_class_persistence"] = clf_metrics(cls_true, cls_persist)
for k in ["stress_class_rf", "stress_class_persistence"]:
    print(k, {a: round(b, 3) for a, b in METRICS[k].items() if a != "confusion"})
print("RF confusion (rows observed, cols predicted; 0 low, 1 moderate, 2 high):")
print(np.array(METRICS["stress_class_rf"]["confusion"]))

# %%
# Cell 20 - SHAP for the forecast model: what drives next-quarter stress?
rf_best, med_best = fitted[best]
Xs = panel.loc[te_mask, SETS_F[best]].fillna(med_best).sample(min(1000, int(te_mask.sum())), random_state=SEED)
sv2 = shap.TreeExplainer(rf_best).shap_values(Xs, check_additivity=False)
plt.figure()
shap.summary_plot(sv2, Xs, show=False, max_display=15)
plt.title(f"SHAP: drivers of stress {CFG['forecast_horizon']} months ahead")
plt.savefig(FIG / "06_shap_stress_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
pd.DataFrame({"feature": SETS_F[best], "mean_abs_shap": np.abs(sv2).mean(0)}).sort_values("mean_abs_shap", ascending=False).to_csv(
    TAB / "06_stress_forecast_shap.csv", index=False)

# %% [markdown]
# ## C. Climate-only model and "non-climatic" stress
# Stress now is predicted from rainfall, temperature, PET and ET of the same period only. What the climate cannot explain
# (the out-of-fold residual) is a proxy for non-climatic drivers such as irrigation and abstraction.

# %%
# Cell 21 - Out-of-fold climate-only model over the whole period (spatial-block CV)
CLIM = ["rain1_z", "rain3_z", "rain6_z", "rain3_z_lag3", "temp_z", "pet_z", "et_z"]
pn = panel[np.isfinite(panel["stress"])].copy()
pn["month_sin_now"] = np.sin(2 * np.pi * (pn["origin"] % 12) / 12)
pn["month_cos_now"] = np.cos(2 * np.pi * (pn["origin"] % 12) / 12)
CLIM_FEATS = CLIM + ["month_sin_now", "month_cos_now"]
oof_clim = cv_predict(lambda: make_rf("reg", max_samples=MS, leaf=20, mf=0.5), pn[CLIM_FEATS],
                      pn["stress"].to_numpy(), pn["block"].to_numpy(), spatial=True)
pn["clim_pred"] = oof_clim
pn["nonclimatic"] = pn["stress"] - pn["clim_pred"]
METRICS["climate_only_model"] = reg_metrics(pn["stress"].to_numpy(), oof_clim)
print("share of stress variance explained by climate (out-of-fold R2): %.2f" % METRICS["climate_only_model"]["r2"])
pn[["pix", "block", "origin", "stress", "clim_pred", "nonclimatic"]].to_parquet(DER / "panel_climate_resid.parquet", index=False)
# Per-pixel TREND of non-climatic stress (z-units per year). The mean is ~0 by construction (anomalies are relative to the
# same period), so a chronic worsening shows up in the trend, not the mean.
wide = pn.pivot(index="origin", columns="pix", values="nonclimatic")
Yv = wide.to_numpy(); Xv = np.where(np.isfinite(Yv), (wide.index.to_numpy()[:, None] / 12.0) * np.ones_like(Yv), np.nan)
xb, yb = np.nanmean(Xv, 0), np.nanmean(Yv, 0)
slope = np.nansum((Xv - xb) * (Yv - yb), 0) / np.maximum(np.nansum((Xv - xb) ** 2, 0), 1e-9)
slope[np.isfinite(Yv).sum(0) < 36] = np.nan
res = pn.groupby("pix")["nonclimatic"].agg(nonclimatic_mean="mean", nonclimatic_sd="std").reset_index()
res = res.merge(pd.DataFrame({"pix": wide.columns, "nonclimatic_trend": slope}), on="pix", how="left")
res.to_csv(TAB / "06_pixel_nonclimatic_stress.csv", index=False)
print("saved residuals")

# %%
# Cell 22 - Forecast skill by target month + observed vs predicted for the hold-out years
tt = panel.loc[te_mask, ["target_idx"]].copy()
tt["obs"], tt["rf"] = y_te, preds[best]
tt["persist"] = panel.loc[te_mask, "stress"].to_numpy()
g = tt.groupby("target_idx")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(g["obs"].mean().to_numpy(), label="observed (mean of sample)"); ax[0].plot(g["rf"].mean().to_numpy(), label="RF forecast")
ax[0].plot(g["persist"].mean().to_numpy(), "--", alpha=0.6, label="persistence")
ax[0].set_xticks(range(0, g.ngroups, 3)); ax[0].set_xticklabels([MONTHS[i] for i in sorted(tt["target_idx"].unique())][::3], rotation=60)
ax[0].set_title("Hold-out period: mean stress score"); ax[0].legend()
ax[1].hexbin(y_te, preds[best], gridsize=40, bins="log", cmap="viridis", mincnt=1); lim = [-3, 4]
ax[1].plot(lim, lim, "r--"); ax[1].set_xlabel("Observed stress"); ax[1].set_ylabel("Forecast stress"); ax[1].set_title(f"{best}: all hold-out rows")
fig.savefig(FIG / "06_stress_forecast_validation.png", dpi=140, bbox_inches="tight")
plt.show()

# %%
# Cell 23 - Save models and all metrics
joblib.dump({"model": rf_vel, "features": FEATS, "median": med_all}, MOD / "rf_subsidence_velocity.joblib", compress=3)
joblib.dump({"model": clf_final, "features": FEATS, "median": med_all}, MOD / "rf_subsidence_class.joblib", compress=3)
joblib.dump({"model": rf_best, "features": SETS_F[best], "median": med_best}, MOD / "rf_stress_forecast.joblib", compress=3)
(TAB / "06_metrics.json").write_text(json.dumps(METRICS, indent=2))
print("saved models and 06_metrics.json in", OUT)

# %% [markdown]
# **Next:** `07_validation_and_coupling`.
