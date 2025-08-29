"""Data cleaning: reads CSVs from data/raw, standardizes schema and saves parquet files.
If CSVs are missing, generates tiny synthetic data to ensure the pipeline runs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.io import load_csv_or_none, write_table, ITEMS, SALES, PROMO, STORES
from src.utils import PROC, RAW, ensure_dirs

RNG = np.random.default_rng(7)

def synthesize():
    # items
    items = pd.DataFrame({
        "Code": [f"I{i:04d}" for i in range(50)],
        "Description": [f"Item {i}" for i in range(50)],
        "Type": RNG.choice(["Type 1","Type 2","Type 3","Type 4"], size=50),
        "Brand": RNG.choice(["Alpha","Beta","Gamma","Delta"], size=50),
        "Size": RNG.choice(["S","M","L"], size=50),
    })
    # stores
    stores = pd.DataFrame({
        "Supermarket No": [f"S{j:03d}" for j in range(10)],
        "Post-code": RNG.integers(10000, 99999, size=10).astype(str)
    })
    # promo
    promo = []
    for code in items['Code'].sample(30, replace=False, random_state=7):
        for s in stores['Supermarket No']:
            promo.append([code, s, RNG.integers(1, 53), RNG.integers(0,2), RNG.integers(0,2), "P1"])
    promo = pd.DataFrame(promo, columns=["Code","Supermarket No","Week","Feature","Display","Province"])

    # sales
    rows = []
    for _ in range(2000):
        code = items['Code'].iloc[RNG.integers(0, len(items))]
        s = stores['Supermarket No'].iloc[RNG.integers(0, len(stores))]
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(RNG.integers(0, 365)))
        week = int(pd.Timestamp(day).isocalendar().week)
        has_promo = ((promo['Code']==code) & (promo['Supermarket No']==s) & (promo['Week']==week)).any()
        base = float(RNG.normal(5, 2))
        uplift = 2.0 if has_promo else 0.0
        units = max(0, base + uplift + float(RNG.normal(0, 1)))
        amount = units * float(RNG.uniform(2.0, 20.0))
        rows.append([code, amount, units, day, "P1", f"C{RNG.integers(1,1000)}", s, f"B{RNG.integers(1,400)}", day.day_name(), "N"])
    sales = pd.DataFrame(rows, columns=["Code","Amount","Units","Time","Province","CustomerId","Supermarket No","Basket","Day","Voucher"])

    items.to_csv(RAW / "Items.csv", index=False)
    stores.to_csv(RAW / "Supermarkets.csv", index=False)
    promo.to_csv(RAW / "Promotion.csv", index=False)
    sales.to_csv(RAW / "Sales.csv", index=False)

def standardize(df):
    df.columns = [c.strip().title() for c in df.columns]
    return df

def main():
    ensure_dirs()
    items = load_csv_or_none(ITEMS)
    sales = load_csv_or_none(SALES)
    promo = load_csv_or_none(PROMO)
    stores = load_csv_or_none(STORES)

    if any(x is None for x in [items, sales, promo, stores]):
        print("⚠️  Raw CSVs missing — generating small synthetic dataset for demo.")
        synthesize()
        items = pd.read_csv(ITEMS)
        sales = pd.read_csv(SALES)
        promo = pd.read_csv(PROMO)
        stores = pd.read_csv(STORES)

    # Basic cleanup
    for df in [items, sales, promo, stores]:
        df.drop_duplicates(inplace=True)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Types
    if "Time" in sales.columns:
        sales["Time"] = pd.to_datetime(sales["Time"], errors="coerce")

    # Save as parquet
    write_table(items, "items")
    write_table(sales, "sales")
    write_table(promo, "promotion")
    write_table(stores, "supermarkets")
    print("✅ Cleaned files written to data/processed/*.csv")

if __name__ == "__main__":
    main()
