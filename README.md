# Supermart Analytics Assignment

This repository scaffold is set up to complete **Task 1** (supervised learning on supermarket transactions) and the **optional Task 2** (maze navigation / RL). Drop your CSVs into `data/raw/` and follow the steps below.

## Quickstart

```bash
# 1) Create a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Open notebooks
jupyter lab  # or: jupyter notebook
```

Place these files in `data/raw/`:
- `Items.csv`
- `Sales.csv`
- `Promotion.csv`
- `Supermarkets.csv`

> Processed/cleaned outputs will be written to `data/processed/` by the scripts.

## Project Structure

```
supermart-assignment/
  data/
    raw/            # put original CSVs here
    processed/      # cleaned/transformed outputs
  notebooks/
    00_eda.ipynb
    01_modeling.ipynb
  scripts/
    data_cleaning.py
    feature_engineering.py
    train_model.py
    generate_insights.py
    maze_model.py        # optional RL task
  report/
    figures/
    report_template.md
  src/
    io_utils.py
    metrics.py
  README.md
  requirements.txt
  .gitignore
```

## Minimal Run (CLI)

```bash
# Clean + validate + output parquet/csv to data/processed
python scripts/data_cleaning.py

# Build features to data/processed/features.parquet
python scripts/feature_engineering.py

# Train a baseline model and save to data/processed/model.joblib
python scripts/train_model.py

# Generate business insights to report/figures and data/processed/insights/**
python scripts/generate_insights.py
```

## Notes
- Keep the code **explainable**; you may be asked to walk through it.
- Start simple (baseline linear/regression) and iterate.
- Treat `01_modeling.ipynb` as your scratchpad for model comparison and charts; keep `scripts/` reproducible.

---

**Generated:** 2025-08-29T04:16:20
