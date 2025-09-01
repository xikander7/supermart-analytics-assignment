import os
import pandas as pd
import re

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def to_snake(s):
        s = str(s).strip()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_").lower()
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    return df

def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    # engine="pyarrow" is fast and consistent for large CSVs if available.
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path)

def save_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)
