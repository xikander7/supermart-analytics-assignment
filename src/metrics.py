"""Metrics helpers for reporting."""
from __future__ import annotations
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

def promotion_uplift(sales_df: pd.DataFrame, promo_flag_col: str, value_col: str) -> float:
    """Compute relative uplift when promotion flag is 1 vs 0."""
    a = sales_df.loc[sales_df[promo_flag_col] == 1, value_col].mean()
    b = sales_df.loc[sales_df[promo_flag_col] == 0, value_col].mean()
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return float((a - b) / b)
