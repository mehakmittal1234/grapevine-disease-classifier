#!/usr/bin/env bash
# End-to-end, reproducible pipeline: data audit/split -> train 3 models -> evaluate -> compare -> Grad-CAM.
# Usage: ./run_pipeline.sh            (from the project folder, after `pip install -r requirements.txt`)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
mkdir -p artifacts/logs

"$PY" -m grapevine.prepare_data 2>&1 | tee artifacts/logs/prepare_data.log
"$PY" -m grapevine.bias_check 2>&1 | tee artifacts/logs/bias_check.log
for model in mobilenet_v2 efficientnet_b0 resnet50; do
  "$PY" -m grapevine.train --model "$model" 2>&1 | tee "artifacts/logs/train_${model}.log"
  "$PY" -m grapevine.evaluate --model "$model" 2>&1 | tee "artifacts/logs/evaluate_${model}.log"
done
"$PY" -m grapevine.compare 2>&1 | tee artifacts/logs/compare.log
"$PY" -m grapevine.gradcam 2>&1 | tee artifacts/logs/gradcam.log
"$PY" -m grapevine.report 2>&1 | tee artifacts/logs/report.log
echo "Done. Launch the app with: streamlit run app.py"
