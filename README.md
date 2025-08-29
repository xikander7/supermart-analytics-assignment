# Supermart Analytics Assignment

A clean, **human-readable** end-to-end pipeline for the supermarket assignment:
- Data cleaning → feature engineering → model training → business insights.
- Optional **maze RL** toy example to showcase broader ML skills.
- Auto-fallback: if raw CSVs are missing, the pipeline generates **small synthetic data**
  so everything still runs and produces charts + a PDF report.

## Repo Structure

```
supermart-analytics-assignment/
  data/
    raw/            # Place Items.csv, Sales.csv, Promotion.csv, Supermarkets.csv here
    processed/      # Auto-generated parquet/feature files
  notebooks/
    00_quick_eda.ipynb        # (placeholder)
  scripts/
    data_cleaning.py
    feature_engineering.py
    train_model.py
    generate_insights.py
    maze_rl.py
  src/
    io.py
    features.py
    modeling.py
    metrics.py
    plots.py
    utils.py
  report/
    report.pdf
    figures/*.png
  requirements.txt
  run_all.sh
  run_all.bat
  README.md
```

## How to Run (Linux/Mac)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/train_model.py
python scripts/generate_insights.py
# optional
python scripts/maze_rl.py
```

## How to Run (Windows PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python scripts\data_cleaning.py
python scripts\feature_engineering.py
python scripts\train_model.py
python scripts\generate_insights.py
# optional
python scripts\maze_rl.py
```

## Expected Inputs

Place the CSVs (exact names) into `data/raw/`:

- `Items.csv` — columns: `Code, Description, Type, Brand, Size`
- `Sales.csv` — columns: `Code, Amount, Units, Time, Province, CustomerId, Supermarket No, Basket, Day, Voucher`
- `Promotion.csv` — columns: `Code, Supermarket No, Week, Feature, Display, Province`
- `Supermarkets.csv` — columns: `Supermarket No, Post-code`

> **No data?** No problem. The pipeline will auto-create tiny synthetic versions so you can demo end-to-end immediately.

## Deliverables
- **report/report.pdf** — auto-generated, includes: problem setup, methodology, model metrics, feature importances, promo lift, store performance, and bonus maze RL.
- **report/figures/*.png** — charts saved for quick review.
- **data/processed/** — cleaned CSV/PKL tables + `features.csv` and `label.csv`.

## Notes for Interview / Client
- The code is **commented** and intentionally straightforward (pandas + scikit-learn + matplotlib).
- Models: baseline + `RandomForestRegressor` (predicts `Units`). Swap target to `Amount` via CLI flag if desired.
- Business insights focus on **promotion lift** and **store ranking**. The implementation is modular so you can
  plug in additional features (e.g., weather, holidays) in `src/features.py`.

---

**Reference:** The repo is aligned to the assignment brief you shared (datasets, supervised learning task, PDF report, and optional RL maze demo).

