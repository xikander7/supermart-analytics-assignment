# Supermart Analytics — Step 1: Data Cleaning (Minimal)

This is a **small, easy first step** so you can run things without the full project.
It cleans the four CSVs and writes standardized Parquet files.

## Folder layout
```text
supermart_step1_cleaning/
  data/
    raw/            # <-- put Item.csv, Sales.csv, Promotion.csv, Supermarkets.csv here
    processed/      # outputs written here
  scripts/
    01_data_audit_and_clean.py
  src/
    io_utils.py
  report/figures/
  models/
  notebooks/
```

## How to run (Windows PowerShell)
1. Open PowerShell at this folder (right-click → "Open in Terminal").
2. Create a virtual env and install deps:
   ```powershell
   python -m venv .venv
   . .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Make Python see `src/` (do this in each new terminal session):
   ```powershell
   $env:PYTHONPATH = (Get-Location).Path
   ```
4. Put these files into `data\raw\` **with the exact names**:
   - `Item.csv`
   - `Sales.csv`
   - `Promotion.csv`
   - `Supermarkets.csv`
5. Run the cleaner:
   ```powershell
   python scripts\01_data_audit_and_clean.py
   ```

## Expected outputs
- `data/processed/items_clean.parquet`
- `data/processed/sales_clean.parquet`
- `data/processed/promotions_clean.parquet`
- `data/processed/supermarkets_clean.parquet`

Console will also print quick quality checks (row counts, nulls, date coverage where applicable).

> If you get `FileNotFoundError`, make sure you **run from the repo root** and the CSVs are inside `data/raw/`.
