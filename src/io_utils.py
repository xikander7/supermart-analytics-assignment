import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def read_csv_safe(name: str, **kwargs) -> pd.DataFrame:
    """
    Reads a CSV from data/raw with basic defaults.
    """
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Expected raw file not found: {path}")
    return pd.read_csv(path, **kwargs)

def write_df(df: pd.DataFrame, name: str, fmt: str = "parquet", **kwargs) -> str:
    """
    Writes a DataFrame to data/processed/ in the chosen format.
    Returns the path string.
    """
    out = PROCESSED_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(out, index=False, **kwargs)
    elif fmt == "csv":
        df.to_csv(out, index=False, **kwargs)
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'")
    return str(out)
