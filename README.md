# Supermart — Simple Edition

## Layout
- `data/` — put `cleaned_master.csv` here (already included if you ran the ETL).
- `notebooks/` — 01 (prepare), 02 (model + insights).
- `report/` — Business report + quick_insights.md.
- `maze/` — optional Q-learning demo.

## Run
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Open the two notebooks in order. The second notebook prints metrics and writes insights to `report/quick_insights.md`.


## `src/` package (important + simple)
- `src/io.py` — load file, detect/build datetime, add time features, promo flag.
- `src/features.py` — build X/y and define numeric vs categorical features.
- `src/modeling.py` — build scikit-learn pipeline and evaluate.
- `src/insights.py` — promotion uplift, top groups, write quick insights.

## Headless run
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_all.py
```
