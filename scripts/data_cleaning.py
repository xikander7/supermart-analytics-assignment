#!/usr/bin/env python3
"""
Data cleaning & normalization.
- Reads Items.csv, Sales.csv, Promotion.csv, Supermarkets.csv from data/raw
- Performs light validation & dtype normalization
- Writes cleaned parquet files to data/processed
"""

import pandas as pd
from src.io_utils import read_csv_safe, write_df

def normalize_items(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip()
    # Optional: size numeric extraction
    if "size" in df.columns:
        # Try to coerce numeric part
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
    return df

def normalize_sales(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in ["amount", "units"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    time_cols = ["time_of_transactions", "transaction_time", "time"]
    for tcol in time_cols:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=False)
            df = df.rename(columns={tcol: "transaction_time"})
            break
    return df

def normalize_promotion(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in ["feature", "display"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def normalize_supermarkets(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "post-code" in df.columns:
        df = df.rename(columns={"post-code": "post_code"})
    return df

def main():
    items = normalize_items(read_csv_safe("Items.csv"))
    sales = normalize_sales(read_csv_safe("Sales.csv"))
    promo = normalize_promotion(read_csv_safe("Promotion.csv"))
    supers = normalize_supermarkets(read_csv_safe("Supermarkets.csv"))

    write_df(items, "items.parquet")
    write_df(sales, "sales.parquet")
    write_df(promo, "promotion.parquet")
    write_df(supers, "supermarkets.parquet")

    print("✅ Cleaned files written to data/processed/*.parquet")

if __name__ == "__main__":
    main()
