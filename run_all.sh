#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)"
echo "PYTHONPATH=$PYTHONPATH"
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/train_model.py
python scripts/generate_insights.py
