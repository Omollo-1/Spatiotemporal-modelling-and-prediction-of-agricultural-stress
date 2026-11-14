#!/usr/bin/env bash
# Runs the whole DEMO pipeline (synthetic data) as a smoke test: 00 -> 00b -> 05 -> 06 -> 07 -> 08.
# Usage: NAIVASHA_DIR=/some/scratch/folder bash tools/run_demo.sh
set -e
export NAIVASHA_DIR="${NAIVASHA_DIR:-$(pwd)}"
mkdir -p "$NAIVASHA_DIR/_executed"
for nb in 00_setup_and_config 00b_demo_synthetic_data 05_build_features 06_random_forest_models 07_validation_and_coupling 08_maps_and_report; do
  echo "== $nb"
  jupyter nbconvert --to notebook --execute "notebooks/$nb.ipynb" --output-dir "$NAIVASHA_DIR/_executed" --ExecutePreprocessor.timeout=3600
done
echo "Demo finished. See $NAIVASHA_DIR/outputs_demo/RESULTS_SUMMARY.md"
