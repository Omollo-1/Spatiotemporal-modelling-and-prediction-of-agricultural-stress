# Naivasha Basin: agricultural stress and land subsidence (2015-2024)

**Spatiotemporal modelling and prediction of agricultural stress and land subsidence using Sentinel-1 InSAR, Sentinel-2 and Random Forest - Lake Naivasha Basin, Kenya.**

A notebook pipeline (one simple script per cell) that runs unchanged on **Google Colab, Jupyter / JupyterLab, VS Code, PyCharm and Spyder**.
It measures ground subsidence (Sentinel-1 InSAR via HyP3 + MintPy), measures vegetation and moisture stress (Sentinel-2 via Earth Engine),
tests whether the two are coupled, and trains validated Random Forest models to map subsidence susceptibility and forecast crop stress.

> **Real-data default.** Notebook 00 is configured for Earth Engine project `operating-rush-505912-s8`; generated data and results are stored in
> `MyDrive/naivasha_project/` when run in Colab. Google authentication is interactive and must be completed by the account that has access to
> this Earth Engine project. Notebook 00b is an optional synthetic demo and is not part of the real-data workflow.
>
> **Validation note.** Earth Engine authentication and the configured project were verified; a live Sentinel-2 query returned 363 scenes over
> the study area in 2020. The full raster downloads and ASF HyP3 processing have not been run here. See `docs/04_troubleshooting.md` if a
> catalog asset or service option has changed.

## The pipeline

| # | Notebook | Needs | What it produces |
|---|---|---|---|
| 00 | `00_setup_and_config` | - | AOI shapefile, `config.json`, analysis grid, interactive OSM preview (run first) |
| 00b | `00b_demo_synthetic_data` | demo mode | synthetic data in the same layout as real data |
| 01 | `01_sentinel2_stress_indices` | Earth Engine | monthly NDVI, NDMI, MSI, NDRE, EVI 2015-2024 |
| 02 | `02_climate_terrain_landcover` | Earth Engine | rainfall, temperature, PET, ET, DEM, slope, clay, lake, land cover |
| 03 | `03_insar_hyp3_download` | Earthdata login | Sentinel-1 interferograms (ASF HyP3), dry-run by default |
| 04 | `04_insar_mintpy_timeseries` | MintPy | vertical velocity (mm/yr) and monthly displacement |
| 05 | `05_build_features` | 01-04 or demo | stress score, trends, distances; pixel table and pixel-month panel |
| 06 | `06_random_forest_models` | 05 | subsidence RF, stress-forecast RF, climate-only RF, SHAP |
| 07 | `07_validation_and_coupling` | 06 | null test, GNSS check, block-bootstrap coupling, hotspots, lags |
| 08 | `08_maps_and_report` | 07 | figures, GIF time-lapse, COGs, web map, `RESULTS_SUMMARY.md` |

## Quick start

**Real-data run.** Run `00` and complete Earth Engine authentication when prompted, then run `01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08`.
Notebook 00 is already set to `MODE = "real"` and `GEE_PROJECT = "operating-rush-505912-s8"`. The optional synthetic workflow is separate:
set `MODE = "demo"` in notebook 00 and run 00b only when you explicitly want test data.
Notebook 03 searches real Sentinel-1 scenes but is intentionally a dry run by default; review the estimated HyP3 credit use before setting
`SUBMIT = True`. Notebook 04 needs completed HyP3 products.
Notebook 03 accepts Earthdata credentials from `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or prompts for both with hidden input. Do not
put credentials in notebook cells or commit them to the repository.

### Google Colab
1. Open `notebooks/00_setup_and_config.ipynb` in Colab (from GitHub or after uploading the repo to Drive).
2. Run cell 1 to mount Drive, then run the notebook. Sign in with the Google account authorized for the Earth Engine project when prompted.
3. All project data and outputs are saved under `MyDrive/naivasha_project/`, so a Colab runtime reset does not remove them. Each notebook installs
   what it needs on its first cell.

### Jupyter / JupyterLab / VS Code
```bash
conda env create -f environment.yml && conda activate naivasha     # or: pip install -r requirements.txt
jupyter lab                                                          # open notebooks/00_setup_and_config.ipynb
```
Data go to `data/` and results to `outputs/` under the project folder. To save them to Google Drive outside Colab, set `NAIVASHA_DIR` to a
Google Drive for Desktop synced folder before starting Jupyter.
Notebook 00 reloads the saved catchment shapefile as the authoritative AOI, writes its geometry to `config.json`, and saves an interactive
OpenStreetMap preview to `outputs/maps/00_study_area_interactive.html`. Analysis rasters stay on the common rectangular grid, with all
pixels outside the AOI masked as nodata when read or written.

### PyCharm / Spyder / any IDE (scripts)
`scripts/*.py` are the same notebooks in "percent" format (`# %%` marks a cell). Open one, run it cell by cell, or `python scripts/05_build_features.py`.

## Folder layout
```
notebooks/   .ipynb files (generated)          scripts/   .py percent scripts (generated)
tools/src/   source of both; edit here, then   python tools/build.py
docs/        methodology, data dictionary, running guide, troubleshooting
data/        raw + derived rasters/tables      outputs/   figures, maps, tables, models
config.json  written by notebook 00
```

## Documentation
* `docs/01_methodology.md` - the science: indices, stress score, InSAR processing, models, validation
* `docs/02_data_dictionary.md` - every file and column
* `docs/03_running_guide.md` - Colab / Jupyter / IDE details, runtimes, memory
* `docs/04_troubleshooting.md` - known pitfalls and fixes
* Full project report: see the Claude Doc "Naivasha Basin: Agricultural Stress and Land Subsidence (2015-2024) - Project Report".

## Honest limitations
* Subsidence vs stress is **association, not proof of causation**; tectonic and volcanic deformation in the rift is not subsidence from abstraction.
* Vertical motion is derived from one or two LOS directions assuming little horizontal motion.
* Greenhouses and dense crops decorrelate in C-band InSAR, so velocity is missing exactly where abstraction is highest; the RF fills the map using drivers.
* Only ~100 independent months exist for the stress forecast, so the forecast forest is deliberately shallow and its skill is judged against climatology.
