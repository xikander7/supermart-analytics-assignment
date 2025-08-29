#!/usr/bin/env python3
"""
Feature engineering:
- Loads cleaned parquet files
- Creates modeling table (joined facts + dims + promo)
- Saves to data/processed/features.parquet
"""
import pandas as pd
from src.io_utils import PROCESSED_DIR, write_df

def main():
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    sales = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    promo = pd.read_parquet(PROCESSED_DIR / "promotion.parquet")
    supers = pd.read_parquet(PROCESSED_DIR / "supermarkets.parquet")

    for df in [items, sales, promo, supers]:
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.strip()
        if "supermarket_no" in df.columns:
            df["supermarket_no"] = df["supermarket_no"].astype(str).str.strip()

    df = sales.merge(items, on="code", how="left")
    if "supermarket_no" in df.columns and "supermarket_no" in promo.columns:
        df = df.merge(promo, on=["code","supermarket_no","province"], how="left", suffixes=("","_promo"))
    if "supermarket_no" in df.columns and "supermarket_no" in supers.columns:
        df = df.merge(supers, on="supermarket_no", how="left", suffixes=("","_store"))

    if "transaction_time" in df.columns:
        dt = pd.to_datetime(df["transaction_time"], errors="coerce")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["weekofyear"] = dt.dt.isocalendar().week.astype("Int64")
        df["dow"] = dt.dt.dayofweek
        df["hour"] = dt.dt.hour

    write_df(df, "features.parquet")
    print("✅ features.parquet created")

if __name__ == "__main__":
    main()
