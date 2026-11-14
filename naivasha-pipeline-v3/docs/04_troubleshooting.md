# Troubleshooting and known risks

| Symptom | Likely cause | Fix |
|---|---|---|
| `config.json not found` | notebook 00 not run, or a different `NAIVASHA_DIR` | run 00; keep the same project folder |
| Earth Engine: "project not registered" / permission | `GEE_PROJECT` wrong | register at code.earthengine.google.com/register; put the id in notebook 00 |
| EE: "Computation timed out" / "Total request size" | tile too large or too heavy | set `TILE = 256` in 01/02; re-run (finished files are skipped) |
| 01: blank months in 2015 | Sentinel-2 starts mid-2015 | expected; baseline uses 2016-2024 |
| 01: Cloud Score+ collection empty for early years | availability differs by period | check `ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")` coverage; fall back to the QA60 band mask |
| 02: an asset id fails | datasets get versioned/renamed | update the id in the notebook (catalog: developers.google.com/earth-engine/datasets) |
| 03: no scenes / wrong frame | `OVERRIDE` needed | check footprints on search.asf.alaska.edu and set `OVERRIDE` |
| 03: HyP3 auth error | Earthdata login or EULA | log in at hyp3-api.asf.alaska.edu once |
| 03: `prepare_insar_job` argument error | hyp3-sdk version changed options | print `help(hyp3.prepare_insar_job)` and adjust the keyword arguments |
| 04: `No module named 'mintpy'` or pip install fails on Windows | notebook kernel uses a Python outside the project Conda environment | create/update the `naivasha` environment with `conda env create -f environment.yml` (or `conda env update -f environment.yml --prune`), then select the `naivasha` kernel in VS Code and restart it |
| 04: MintPy cannot find files | HyP3 file naming changed | check the `*_unw_phase.tif`, `*_corr.tif`, `*_dem.tif`, `*_lv_theta.tif`, `*_lv_phi.tif` names; edit `CONFIG_TEMPLATE` |
| 04: "products differ in size" | pairs from different frames | use one track/frame per direction (notebook 03) |
| 04: ramp across the whole velocity map | reference point or orbit error | set `REF_LALO` to a stable coherent point; try GACOS |
| 05: out of memory | grid too fine | use `RES = 200` or a smaller AOI |
| 06: SHAP is slow | large forest | lower `RF_TREES` or the SHAP sample size |
| Negative skill vs climatology in 06 | the honest result for short records | report it; do not tune on the test years |
| Spearman near zero in 07 | wrong stress metric or real absence of coupling | use trend metrics; check hotspot map; absence is a valid finding |

## Known limitations of the code as delivered
* Notebooks 01-03 were not executed against live Earth Engine / ASF services. 04 was exercised on simulated MintPy output, not real interferograms.
* Demo-mode numbers come from invented data; ignore them scientifically.
* `SPI` is a standardised anomaly (not gamma-fitted). `AG_MIN_FRAC`, class breaks and the stress definition are modelling choices, documented and adjustable.
