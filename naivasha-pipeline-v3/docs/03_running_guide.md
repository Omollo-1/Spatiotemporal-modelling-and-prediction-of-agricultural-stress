# Running guide

## Where it runs
| Platform | How | Notes |
|---|---|---|
| Google Colab | open `notebooks/00_setup_and_config.ipynb` from GitHub or Drive | cell 1 mounts Drive; data and results live in `MyDrive/naivasha_project/`. Free Colab has ~12 GB RAM: fine at 100 m for the default AOI. |
| Jupyter / JupyterLab | `conda env create -f environment.yml` | run from the repo; data in `data/` |
| VS Code | open `notebooks/*.ipynb` (Jupyter extension) or `scripts/*.py` (Run Cell) | select the `naivasha` interpreter |
| PyCharm / Spyder | open `scripts/*.py` | `# %%` cells run individually; or run the whole file |
| Anywhere | `NAIVASHA_DIR=/path python scripts/05_build_features.py` | override the project folder with the env var |

Cell 1 of every notebook detects the platform and mounts Google Drive in Colab. `PROJECT_DIR` is `MyDrive/naivasha_project/` in Colab; locally,
set `NAIVASHA_DIR` to a Google Drive for Desktop synced folder to persist files there. Inputs, intermediate data, and outputs all live beneath
that directory. Cell 2 (from notebook 01 onward) loads `config.json`.

Notebook 00 defaults to real data and Earth Engine project `operating-rush-505912-s8`. It retrieves the Lake Naivasha catchment from the
HydroBASINS level-8 Earth Engine collection, saves and reloads `data/ancillary/naivasha_catchment.shp`, and writes that polygon geometry to
`config.json`. It also saves an interactive OpenStreetMap AOI preview to `outputs/maps/00_study_area_interactive.html`. The grid is
rectangular for raster storage, but pixels outside this polygon are masked as nodata on raster reads and writes. Run notebook 00 first and
complete the Google authentication prompt. Then run 01-08 in order; do not run the optional synthetic-only 00b notebook for a real analysis. Notebook 03 only
searches and estimates HyP3 processing credits unless you explicitly set `SUBMIT = True`; notebook 04 requires completed HyP3 products.

## Order and approximate runtime (default AOI, 100 m)
| Notebook | Time | Peak RAM |
|---|---|---|
| 00, 00b | < 1 min | < 1 GB |
| 01 | 30-90 min (Earth Engine, resumable) | < 2 GB |
| 02 | 15-40 min (resumable) | < 2 GB |
| 03 | minutes to order; HyP3 needs 20-60 min per job batch | small |
| 04 | tens of minutes per track | 4-8 GB |
| 05 | 5-15 min | 3-6 GB |
| 06 | 15-60 min depending on cores (Colab has 2) | 3-6 GB |
| 07, 08 | < 10 min | 2 GB |
Demo mode (500 m grid): the whole chain takes about 6-8 minutes on a laptop. Demo outputs are synthetic and must not be used as research results.

## Reducing cost
* Coarser grid (`RES = 200`) quarters memory and time; set it in notebook 00 and re-run from 01.
* Fewer pixels in the panel (`SAMPLE_PIXELS`), fewer trees (`RF_TREES`).
* Fewer InSAR pairs: shorter period or `MAX_NEIGHBOURS = 1` in notebook 03.

## Accounts needed (real mode)
1. Google account authorized to use the Earth Engine-enabled Cloud project `operating-rush-505912-s8` (https://earthengine.google.com).
2. NASA Earthdata login (https://urs.earthdata.nasa.gov) and ASF HyP3 access (accept the EULA once).

Notebook 03 reads `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` from the process environment when both are set; otherwise, it prompts for both
using hidden input. Never paste credentials into notebook source or commit them.

## Rebuilding notebooks
Edit `tools/src/*.py` (and `tools/snippets/*.py` for shared cells), then `python tools/build.py` regenerates `notebooks/` and `scripts/`.
Smoke test the demo chain: `NAIVASHA_DIR=/tmp/naivasha_test bash tools/run_demo.sh`.
