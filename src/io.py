import pandas as pd
from pathlib import Path
from .utils import RAW, PROC, ensure_dirs

ITEMS = RAW / "Items.csv"
SALES = RAW / "Sales.csv"
PROMO = RAW / "Promotion.csv"
STORES = RAW / "Supermarkets.csv"

# Portable IO: prefer CSV/PKL to avoid optional parquet deps
def write_table(df: pd.DataFrame, name: str):
    ensure_dirs()
    name = name.replace(".parquet", "").replace(".csv", "")
    p_csv = PROC / f"{name}.csv"
    p_pkl = PROC / f"{name}.pkl"
    df.to_csv(p_csv, index=False)
    df.to_pickle(p_pkl)
    return p_csv

def read_table(name: str) -> pd.DataFrame:
    name = name.replace(".parquet", "").replace(".csv", "").replace(".pkl","")
    p_csv = PROC / f"{name}.csv"
    p_pkl = PROC / f"{name}.pkl"
    if p_csv.exists():
        return pd.read_csv(p_csv)
    elif p_pkl.exists():
        return pd.read_pickle(p_pkl)
    else:
        raise FileNotFoundError(f"No table found for {name} (csv/pkl).")

def load_csv_or_none(path: Path):
    return pd.read_csv(path) if path.exists() else None
