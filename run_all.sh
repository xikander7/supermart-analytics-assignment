#!/usr/bin/env bash
set -e
export PYTHONPATH=$(pwd)
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/train_model.py
python scripts/generate_insights.py
python scripts/maze_rl.py
