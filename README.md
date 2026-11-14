# Spatiotemporal Modelling of Agricultural Stress

An open, reproducible geospatial pipeline for investigating agricultural stress and land subsidence in the Lake Naivasha Basin, Kenya (2015–2024). It combines Sentinel-2 vegetation observations, climate and terrain data, Sentinel-1 InSAR, and spatially aware machine-learning models.

> **Interpretation:** The analysis tests spatial and temporal associations; it does not establish that groundwater abstraction causes observed subsidence or vegetation stress. See the [limitations](naivasha-pipeline-v3/README.md#honest-limitations) before interpreting results.

## What the project does

- Builds monthly vegetation and moisture indicators from Sentinel-2, including NDVI, NDMI, MSI, NDRE, and EVI.
- Adds climate, terrain, land-cover, hydrology, and optional InSAR-derived features.
- Processes Sentinel-1 interferograms with ASF HyP3 and MintPy to estimate ground-motion time series.
- Trains and evaluates Random Forest models for subsidence susceptibility and future agricultural stress.
- Evaluates model skill with spatial and temporal holdouts, and examines stress–subsidence coupling.
- Produces maps, figures, tables, and a results summary.

## Repository contents

The application is currently organized under [`naivasha-pipeline-v3/`](naivasha-pipeline-v3/):

| Path | Purpose |
|---|---|
| [`notebooks/`](naivasha-pipeline-v3/notebooks/) | Ordered, interactive analysis notebooks |
| [`scripts/`](naivasha-pipeline-v3/scripts/) | Python scripts corresponding to the notebooks |
| [`tools/src/`](naivasha-pipeline-v3/tools/src/) | Source used to generate notebooks and scripts |
| [`docs/`](naivasha-pipeline-v3/docs/) | Methodology, data dictionary, running guide, and troubleshooting |
| `data/`, `data_demo/`, `outputs/`, `outputs_demo/` | Local input, demo, and generated files; not included in this repository |

## Getting started

### Requirements

- Python with the dependencies in `naivasha-pipeline-v3/environment.yml` (Conda) or `naivasha-pipeline-v3/requirements.txt` (pip)
- For real-data processing: access to Google Earth Engine and an authorized Earth Engine Cloud project
- For InSAR processing: a NASA Earthdata account and ASF HyP3 access; MintPy is needed for time-series analysis

### Install

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Omollo-1/Spatiotemporal-modelling-and-prediction-of-agricultural-stress.git
cd Spatiotemporal-modelling-and-prediction-of-agricultural-stress/naivasha-pipeline-v3
conda env create -f environment.yml
conda activate naivasha
jupyter lab
```

Alternatively, install with `pip install -r requirements.txt` in your chosen Python environment. Open `notebooks/00_setup_and_config.ipynb` and run it first to create the study-area configuration and analysis grid. Complete the interactive Earth Engine sign-in when prompted, then run notebooks `01` through `08` in order.

### Synthetic demo

The optional synthetic workflow can be run without downloading the real satellite data:

1. In `notebooks/00_setup_and_config.ipynb`, set `MODE = "demo"`.
2. Run notebook `00b_demo_synthetic_data.ipynb`.
3. Run notebooks `05` through `08` to build features, fit models, validate, and generate maps.

Demo results are synthetic and must not be presented as research findings. For more detail, see the [running guide](naivasha-pipeline-v3/docs/03_running_guide.md).

## Data and credentials

Large input datasets, generated results, and `config.json` are excluded by the project’s `.gitignore`. Run the setup notebook to create configuration and data directories locally. Do not commit credentials or private data.

The InSAR download notebook defaults to a dry run. Review estimated HyP3 processing costs before enabling job submission. Credentials should be provided through environment variables or the notebook’s hidden prompts, never hard-coded into source files.

## Analysis sequence

| Step | Notebook | Main output |
|---|---|---|
| Setup | `00_setup_and_config` | Study area, configuration, analysis grid |
| Demo (optional) | `00b_demo_synthetic_data` | Synthetic inputs for testing |
| Sentinel-2 | `01_sentinel2_stress_indices` | Monthly vegetation and moisture indices |
| Climate and land | `02_climate_terrain_landcover` | Climate, terrain, and land-cover layers |
| InSAR | `03_insar_hyp3_download` → `04_insar_mintpy_timeseries` | Interferograms and ground-motion time series |
| Features and models | `05_build_features` → `06_random_forest_models` | Analysis tables and trained models |
| Validation and reporting | `07_validation_and_coupling` → `08_maps_and_report` | Validation results, coupling analysis, maps, and report |

## Documentation

- [Project README and detailed pipeline guide](naivasha-pipeline-v3/README.md)
- [Methodology](naivasha-pipeline-v3/docs/01_methodology.md)
- [Data dictionary](naivasha-pipeline-v3/docs/02_data_dictionary.md)
- [Running guide](naivasha-pipeline-v3/docs/03_running_guide.md)
- [Troubleshooting](naivasha-pipeline-v3/docs/04_troubleshooting.md)

To regenerate notebooks and scripts after editing shared sources, run `python tools/build.py` from `naivasha-pipeline-v3/`.
