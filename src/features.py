"""Feature engineering helpers."""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["year_week"] = d.dt.strftime("%Y-%U")
    return df

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def agg_daily_store_sales(df: pd.DataFrame, date_col: str, store_col: str, value_col: str) -> pd.DataFrame:
    grp = (df
        .groupby([store_col, pd.Grouper(key=date_col, freq="D")])[value_col]
        .sum()
        .reset_index())
    grp.rename(columns={value_col: "daily_store_sales", date_col: "date"}, inplace=True)
    return grp
