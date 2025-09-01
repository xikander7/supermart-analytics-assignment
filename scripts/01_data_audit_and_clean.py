import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from io_utils import ensure_dir, standardize_columns, read_csv_safe, save_parquet

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")

FILES = {
    "items":        os.path.join(RAW_DIR, "Item.csv"),
    "sales":        os.path.join(RAW_DIR, "Sales.csv"),
    "promotions":   os.path.join(RAW_DIR, "Promotion.csv"),
    "supermarkets": os.path.join(RAW_DIR, "Supermarkets.csv"),
}

def _print_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def _quick_info(df: pd.DataFrame, name: str, date_col: str | None=None):
    print(f"[{name}] rows: {len(df):,} | cols: {len(df.columns)}")
    nulls = df.isna().mean().sort_values(ascending=False)
    print("Top nulls (%):")
    print((nulls.head(5)*100).round(2).to_string())
    if date_col and date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        print(f"Date coverage for {date_col}: min={df[date_col].min()} | max={df[date_col].max()}")

def clean_items(path: str) -> pd.DataFrame:
    df = read_csv_safe(path)
    df = standardize_columns(df)
    # Normalize column names
    rename_map = {
        "code": "item_code",
        "description": "item_description",
        "type": "item_type",
        "brand": "brand",
        "size": "size",
    }
    df = df.rename(columns=rename_map)
    # Basic trims
    for col in ["item_description", "item_type", "brand", "size"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    # Drop exact duplicates
    df = df.drop_duplicates()
    return df

def clean_sales(path: str) -> pd.DataFrame:
    df = read_csv_safe(path)
    df = standardize_columns(df)
    # Expected original columns include: code, amount, units, time, province, customer id,
    # supermarket no, basket, day, voucher
    rename_map = {
        "code": "item_code",
        "customer_id": "customer_id",
        "customerid": "customer_id",
        "supermarket_no": "supermarket_no",
        "supermarket_number": "supermarket_no",
        "supermarket": "supermarket_no",
        "province": "province",
        "amount": "amount",
        "units": "units",
        "time": "transaction_time",
        "basket": "basket_id",
        "day": "day",
        "voucher": "voucher",
    }
    df = df.rename(columns=rename_map)

    # Parse transaction_time
    if "transaction_time" in df.columns:
        # Try ISO / common formats
        def parse_dt(s):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
                try:
                    return pd.to_datetime(s, format=fmt, errors="coerce")
                except Exception:
                    pass
            return pd.to_datetime(s, errors="coerce")
        df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
    else:
        # If missing, create a placeholder date (not ideal, but prevents crashes)
        df["transaction_time"] = pd.NaT

    # Types: numeric coercion
    for col in ["amount", "units", "supermarket_no"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Province normalization
    if "province" in df.columns:
        df["province"] = df["province"].astype(str).str.strip().str.upper()

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Optional derived columns
    if "transaction_time" in df.columns:
        df["year_week"] = df["transaction_time"].dt.strftime("%Y-%U")
        df["week"] = df["transaction_time"].dt.isocalendar().week.astype("Int64")
        df["year"] = df["transaction_time"].dt.year.astype("Int64")

    return df

def clean_promotions(path: str) -> pd.DataFrame:
    df = read_csv_safe(path)
    df = standardize_columns(df)
    rename_map = {
        "code": "item_code",
        "supermarket_no": "supermarket_no",
        "week": "week",
        "feature": "feature",
        "display": "display",
        "province": "province",
    }
    df = df.rename(columns=rename_map)
    # Types
    for col in ["supermarket_no", "week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "province" in df.columns:
        df["province"] = df["province"].astype(str).str.strip().str.upper()
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def clean_supermarkets(path: str) -> pd.DataFrame:
    df = read_csv_safe(path)
    df = standardize_columns(df)
    rename_map = {
        "supermarket_no": "supermarket_no",
        "postcode": "postcode",
        "postal_code": "postcode",
    }
    df = df.rename(columns=rename_map)
    # Basic string trims
    if "postcode" in df.columns:
        df["postcode"] = df["postcode"].astype(str).str.strip()
    # Types
    if "supermarket_no" in df.columns:
        df["supermarket_no"] = pd.to_numeric(df["supermarket_no"], errors="coerce")
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def main():
    _print_header("Supermart | Step 1: Data Cleaning")
    for key, path in FILES.items():
        if not os.path.exists(path):
            print(f"ERROR: Missing required file: {path}")
    # Proceed and let read_csv_safe raise if truly missing
    items = clean_items(FILES["items"])
    _quick_info(items, "items")
    save_parquet(items, os.path.join(OUT_DIR, "items_clean.parquet"))

    sales = clean_sales(FILES["sales"])
    _quick_info(sales, "sales", date_col="transaction_time")
    save_parquet(sales, os.path.join(OUT_DIR, "sales_clean.parquet"))

    promos = clean_promotions(FILES["promotions"])
    _quick_info(promos, "promotions")
    save_parquet(promos, os.path.join(OUT_DIR, "promotions_clean.parquet"))

    stores = clean_supermarkets(FILES["supermarkets"])
    _quick_info(stores, "supermarkets")
    save_parquet(stores, os.path.join(OUT_DIR, "supermarkets_clean.parquet"))

    print("\n✅ Done. Cleaned parquet files written to data/processed/.")

if __name__ == "__main__":
    # Allow running from anywhere by forcing CWD to script's parent if needed
    # But we still recommend running from repo root.
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        print("Make sure Item.csv, Sales.csv, Promotion.csv, Supermarkets.csv are in data/raw/.")
        sys.exit(1)
