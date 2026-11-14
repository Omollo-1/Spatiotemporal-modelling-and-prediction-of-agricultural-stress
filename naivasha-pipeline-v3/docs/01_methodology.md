# Methodology

## 1. Research questions
1. Where and how fast is the land subsiding (vertical mm/yr) and does it line up with abstraction hotspots?
2. How did vegetation and moisture stress evolve 2015-2024, and how much is climate-driven?
3. Do subsidence and agricultural stress covary in space and time, and with what lag?
4. Can Random Forest predict subsidence susceptibility and future stress, and which drivers matter?

## 2. Study area and grid
Real mode uses the Lake Naivasha catchment dissolved from HydroBASINS level-8 units sharing the outlet basin; setup saves and reloads this geometry
from `data/ancillary/naivasha_catchment.shp`. The `[36.20, -1.05, 36.60, -0.55]` box is used only for the synthetic demo. Everything is resampled
to one UTM 37S grid (`EPSG:32737`, 100 m) so pixels align exactly. The grid is rectangular for raster storage, but pixels outside the AOI shapefile are
masked as nodata throughout analysis.

## 3. Sentinel-2 stress (notebooks 01, 05)
* **Product:** `COPERNICUS/S2_HARMONIZED` (L1C TOA) for the full period, cloud-masked with Cloud Score+ (`cs_cdf >= 0.6`). L2A is not used because coverage
  over Africa is incomplete before about 2018 and mixing products would create an artificial step in the series.
* **Composites:** monthly per-pixel median of clear observations, then area-mean to 100 m. Sentinel-2 only starts mid-2015, so Jan-Jun 2015 are empty
  and 2015-2017 are sparser (one satellite until Sentinel-2B in 2017); the climatology therefore uses 2016-2024 (`BASELINE_YEARS`).
* **Indices:** NDVI, NDMI, MSI (B11/B8, higher = drier), NDRE, EVI.
* **Gap filling:** linear interpolation across at most 2 missing months, only between valid observations. Longer gaps stay missing.
* **Anomaly:** `z = (x - mean_m) / sd_m` where mean and sd are for the same calendar month over the baseline years.
* **Stress score:** `S = mean(-z_NDVI, -z_NDMI, +z_MSI)`. Positive = more stressed than normal. Classes: low < 0.5, moderate 0.5-1.0, high >= 1.0 (adjustable).
* **Important consequence:** because anomalies are relative to the same period, the *mean* stress of every pixel is about 0 and a slow chronic decline is partly
  absorbed into the baseline. Chronic stress therefore shows up in the **trend** (Theil-Sen slope with Mann-Kendall p-value of annual mean stress) and in the share of
  months in high stress. The coupling tests in notebook 07 use the trend metrics as primary.
* **Climate drivers:** CHIRPS rainfall and ERA5-Land temperature, potential and actual evaporation, as monthly z-scores and 3/6-month rainfall z-scores
  (a standardised-anomaly proxy for SPI; a gamma-fitted SPI is a drop-in upgrade).

## 4. Land subsidence (notebooks 03, 04)
* **Data:** Sentinel-1 IW SLC, ascending and descending, one track/frame each (chosen by scene count; check footprints).
* **Interferograms:** ASF HyP3 `INSAR_GAMMA`, 20x4 looks (about 80 m), with DEM and look vectors. Network: each scene to its next two scenes within 36 days.
* **Time series:** MintPy SBAS: coherence-based network, bridging unwrapping-error correction, weighted inversion, DEM-error correction, height-correlation tropospheric correction,
  linear deramp. Optionally use GACOS/ERA5 corrections for better atmosphere removal.
* **Vertical motion:** `v_up = v_LOS / cos(incidence)`, assuming negligible horizontal motion. With two tracks the vertical estimates are averaged and their difference is reported.
  Treat this as vertical-dominant, not exact; use MintPy `asc_desc2horz_vert.py` for a full decomposition.
* **Sentinel-1B** failed in December 2021: revisit goes from 6 to 12 days overall (12 to 24 days for a single satellite over Kenya), so the network thins after 2021.
* **Decorrelation:** greenhouses, dense crops and water decorrelate; the velocity map has holes there. The RF later predicts those pixels from drivers.
* **Natural vs human signal:** subsidence near Olkaria geothermal and uplift/subsidence around Longonot can be geothermal or volcanic. Interpret residuals after considering these.

## 5. Features (notebook 05)
| Group | Features |
|---|---|
| Terrain, soil, hydrology | elevation, slope, clay, log upstream area |
| Land cover | cropland, grass, tree, built, bare fractions (ESA WorldCover) |
| Distances | lake, Olkaria, Longonot, Naivasha town, nearest borehole (+ boreholes within 5 km, optional), geology code (optional) |
| Vegetation/climate summaries | mean and SD of NDVI/NDMI, trends, stress mean/trend/high-share, mean and trend of rain, ET, temperature |
| Dynamic (panel) | stress now and lagged 1-2 months, z-NDVI/NDMI/MSI, rain 1/3/6-month z, rain lag 3, temperature/PET/ET z, month sin/cos of the target month |

## 6. Models (notebook 06)
* **A. Subsidence velocity:** RandomForestRegressor on pixels with valid InSAR. Feature set A = drivers only; B = drivers + vegetation/climate summaries. Severity classifier
  (stable > -5, moderate -5 to -15, severe < -15 mm/yr) with balanced class weights. Final model predicts every land pixel, including decorrelated ones.
* **B. Stress forecast:** target = stress score `H` months ahead (default 3). Inputs only use information up to month t. Train on targets before 2023, test 2023-2024.
  Feature sets F1 dynamics only, F2 + drivers, F3 + subsidence velocity (does knowing the ground is sinking improve the forecast?). Baselines: climatology (0) and persistence.
  Comparator: histogram gradient boosting. The forest is deliberately **shallow and coarse** because only about 100 independent months exist.
* **C. Climate-only model:** predicts same-month stress from climate alone with spatial out-of-fold predictions. Its R2 is the share of stress variance explained by climate;
  the residual is "non-climatic stress", a proxy for irrigation and abstraction effects.

## 7. Validation (notebooks 06, 07)
* **Spatial-block CV** (block = 3 km, `GroupKFold`): neighbouring pixels never cross the train/test split. The notebook also reports ordinary random k-fold to show the inflation.
* **Temporal hold-out:** train before 2023, test 2023-2024. **Strict test:** future years AND 20% of blocks never seen in training.
* **Skill score** `1 - RMSE_model^2 / RMSE_climatology^2`: positive means better than a "normal month" forecast.
* **Metrics:** RMSE, MAE, R2; macro-F1, Cohen's kappa, confusion matrix for classes.
* **Interpretation:** permutation importance on unseen blocks; SHAP for direction and interactions.
* **InSAR uncertainty:** stable-area null test (robust SD of velocity over flat ground far from lake, geothermal field and volcano) sets the significance threshold (-2 sigma, but not milder than -5 mm/yr);
  optional comparison against GNSS/levelling (`gnss.csv`).
* **Coupling:** Spearman correlation of velocity with stress metrics, 95% CI by **spatial-block bootstrap**; hotspot overlap vs chance; lag correlation (+/-6 months) with an effective
  sample size correction for autocorrelation (Bretherton et al., 1999).

## 8. Circularity and leakage guards
* NDVI is never both feature and label for the same pixel and month: the forecast target is stress at t+H; stress at t is an input (a legitimate persistence feature) and is benchmarked against persistence.
* The subsidence model does not use subsidence-derived features; vegetation features are an explicit, separate feature set so their added value can be judged.
* No future climate enters the forecast. Medians for missing-value imputation are computed on training folds only.

## 9. What would strengthen the study
GNSS/levelling data (KenGen, universities), WRA borehole abstraction records, lake level (DAHITI/Hydroweb), farm-polygon and greenhouse maps, GACOS atmospheric corrections,
a second InSAR track pair, and a literature review of published deformation at Olkaria and Longonot to separate natural from human signals.
