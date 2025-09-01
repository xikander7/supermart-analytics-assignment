"""I/O utilities for the Supermart project."""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

def load_config(path: str | Path) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    LOGGER.info("Config loaded from %s", path)
    return cfg

def read_csv_smart(path: str | Path) -> pd.DataFrame:
    """Read CSV with common options and robust date parsing."""
    path = Path(path)
    df = pd.read_csv(path)
    LOGGER.info("Read %s: %s rows, %s cols", path.name, len(df), len(df.columns))
    return df

def ensure_datetime(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Coerce first matching column among *candidates* to datetime."""
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                return df
    return df

def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    LOGGER.info("Wrote parquet -> %s", path)

def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote CSV -> %s", path)
