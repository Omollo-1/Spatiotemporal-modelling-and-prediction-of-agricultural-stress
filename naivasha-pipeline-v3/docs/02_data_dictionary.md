# Data dictionary

All rasters share the project grid (CRS, origin and cell size in `config.json`). Nodata = -9999 on disk, NaN in memory. "12 bands" = one per month.

## Rasters (`data/`)
| File | Bands | Unit | Source / notebook |
|---|---|---|---|
| `s2/NDVI_<yr>.tif`, `NDMI`, `MSI`, `NDRE`, `EVI` | 12 | unitless | 01 |
| `climate/rain_mm_<yr>.tif` | 12 | mm per month | 02 CHIRPS |
| `climate/temp_c_<yr>.tif` | 12 | deg C | 02 ERA5-Land |
| `climate/pet_mm_<yr>.tif`, `et_mm_<yr>.tif` | 12 | mm per month | 02 ERA5-Land (sign flipped to positive) |
| `static/elevation_m.tif`, `slope_deg.tif` | 1 | m, degrees | 02 SRTM |
| `static/clay_pct.tif` | 1 | % | 02 OpenLandMap |
| `static/upa_km2.tif` | 1 | km2 | 02 MERIT Hydro |
| `static/lake_occurrence.tif` | 1 | % of time water | 02 JRC GSW (>= 50 treated as lake) |
| `static/landcover_fractions.tif` | 7 | fraction 0-1 (crop, grass, tree, shrub, built, bare, water) | 02 ESA WorldCover |
| `insar/velocity_vertical_mm_yr.tif` | 1 | mm/yr, negative = sinking | 04 |
| `insar/velocity_std_mm_yr.tif` | 1 | mm/yr | 04 |
| `insar/disp_vertical_monthly_mm.tif` | months | mm, relative to first epoch | 04 |

## Optional inputs (`data/ancillary/`)
| File | Columns | Used for |
|---|---|---|
| `boreholes.csv` | `lon, lat` (+ any extra) | distance to nearest borehole, borehole count within 5 km |
| `gnss.csv` | `lon, lat, vel_up_mm_yr` | independent validation of InSAR velocity |
| `geology.gpkg` | polygons with text column `unit` | integer `geology_code` feature |

## Derived tables (`data/derived/`)
| File | Content |
|---|---|
| `stress_score.npy` | float32 `[months, rows, cols]` stress score |
| `static_pixels.parquet` | one row per land pixel: `pix, row, col, x, y, block, vel, vel_std, is_ag`, terrain/soil/land-cover/distance features, vegetation and climate summaries (`*_mean`, `*_trend`, `stress_frac_high`) |
| `panel_stress.parquet` | one row per sampled pixel x origin month: dynamic features, static features, `target` (stress at `target_idx`), `target_year`, `block` |
| `panel_climate_resid.parquet` | out-of-fold climate-only prediction and residual (`nonclimatic`) per pixel-month |
| `feature_meta.json` | lists of feature names per group, horizon, sizes |

## Outputs (`outputs/`)
| Folder | Content |
|---|---|
| `maps/` | `00_study_area_interactive.html` (setup AOI + OSM basemap), `pred_velocity_mm_yr`, `pred_subsidence_class` (0 stable, 1 moderate, 2 severe), `pred_p_severe`, `oof_velocity_mm_yr`, `hotspot_classes` (0 neither, 1 subsiding, 2 stressed, 3 both), `cog/`, `stress_timelapse.gif`, `naivasha_interactive.html` |
| `figures/` | PNGs from every notebook |
| `tables/` | CV metrics, forecast metrics, importances, SHAP tables, coupling table, lag correlations, `06_metrics.json`, `07_results.json` |
| `models/` | `rf_subsidence_velocity`, `rf_subsidence_class`, `rf_stress_forecast` (joblib dict: model, features, median) |
| `RESULTS_SUMMARY.md` | auto-written summary with the real numbers |

## Key variable definitions
* `stress` : mean of -z(NDVI), -z(NDMI), +z(MSI). `stress_trend` : Theil-Sen slope of annual mean stress (z/yr). `stress_frac_high` : share of months with stress >= 1.
* `vel` : InSAR vertical velocity (mm/yr). `block` : spatial block id for cross-validation.
* `nonclimatic` : stress minus out-of-fold climate-only prediction. `nonclimatic_trend` : its per-pixel slope (z/yr).
