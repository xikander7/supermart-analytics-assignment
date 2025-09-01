#!/usr/bin/env python
\"\"\"Clean & normalize raw datasets for Supermart project.\"\"\"
import logging, sys
from pathlib import Path
import pandas as pd
from src.io import load_config, read_csv_smart, ensure_datetime, write_parquet, write_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a")
    ],
)
LOGGER = logging.getLogger("01_data_cleaning")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

def detect_columns(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for wanted in candidates:
        for c in cols_lower:
            if c == wanted.lower():
                return cols_lower[c]
    return None

def main():
    cfg = load_config("config.yaml")
    raw = Path(cfg["paths"]["data_raw"])
    out = Path(cfg["paths"]["data_processed"])
    out.mkdir(parents=True, exist_ok=True)

    sales = standardize_columns(read_csv_smart(raw / cfg["files"]["sales"]))
    items = standardize_columns(read_csv_smart(raw / cfg["files"]["items"]))
    promos = standardize_columns(read_csv_smart(raw / cfg["files"]["promotions"]))
    stores = standardize_columns(read_csv_smart(raw / cfg["files"]["supermarkets"]))

    # Date handling
    for df, candidates in [
        (sales, ["date", "transaction_date", "order_date"]),
        (promos, ["start_date", "promo_start", "start"]),
    ]:
        ensure_datetime(df, candidates)

    # Heuristic column mapping
    col_map = {}
    for col, wants in [
        ("date", ["date", "transaction_date"]),
        ("store_id", ["store_id", "store", "supermarket_id"]),
        ("item_id", ["item_id", "sku", "product_id"]),
        ("quantity", ["quantity", "qty", "units"]),
        ("price", ["price", "unit_price"]),
        ("amount", ["amount", "sales", "revenue", "total"]),
    ]:
        found = detect_columns(sales, wants)
        if found:
            col_map[col] = found

    # Derive amount if missing
    sales = sales.copy()
    if "amount" not in col_map:
        if "quantity" in col_map and "price" in col_map:
            sales["amount"] = pd.to_numeric(sales[col_map["quantity"]], errors="coerce") * \
                              pd.to_numeric(sales[col_map["price"]], errors="coerce")
            col_map["amount"] = "amount"

    required = ["date", "store_id", "item_id", "amount"]
    for r in required:
        if r not in col_map:
            LOGGER.warning("Could not find required column: %s", r)

    norm_sales = sales.rename(columns={v: k for k, v in col_map.items()})
    if "date" in norm_sales.columns:
        norm_sales["date"] = pd.to_datetime(norm_sales["date"], errors="coerce")

    # Items normalization
    def colmap_from(df, mapping_pairs):
        cm = {}
        for col, wants in mapping_pairs:
            for w in wants:
                if w in [c.lower() for c in df.columns]:
                    cm[col] = [c for c in df.columns if c.lower()==w][0]
                    break
        return cm

    items_colmap = colmap_from(items, [
        ("item_id", ["item_id","sku","product_id"]),
        ("category", ["category"]),
        ("subcategory", ["subcategory","sub_category"]),
        ("brand", ["brand"]),
        ("item_name", ["item_name","name","title"]),
    ])
    items_norm = items.rename(columns={v: k for k, v in items_colmap.items()})

    promos_colmap = colmap_from(promos, [
        ("item_id", ["item_id","sku","product_id"]),
        ("store_id", ["store_id","store","supermarket_id"]),
        ("start_date", ["start_date","promo_start","start"]),
        ("end_date", ["end_date","promo_end","end"]),
        ("discount", ["discount","discount_pct","pct_off"]),
        ("promo_flag", ["promo_flag","is_promo","promotion"]),
    ])
    promos_norm = promos.rename(columns={v: k for k, v in promos_colmap.items()})
    if "promo_flag" not in promos_norm.columns:
        promos_norm["promo_flag"] = 1

    stores_colmap = colmap_from(stores, [
        ("store_id", ["store_id","store","supermarket_id"]),
        ("city", ["city"]),
        ("state", ["state","region"]),
        ("store_name", ["store_name","name"]),
    ])
    stores_norm = stores.rename(columns={v: k for k, v in stores_colmap.items()})

    write_parquet(norm_sales, out / "sales.parquet")
    write_parquet(items_norm, out / "items.parquet")
    write_parquet(promos_norm, out / "promotions.parquet")
    write_parquet(stores_norm, out / "stores.parquet")

    write_csv(norm_sales.head(50000), out / "sales_sample.csv")
    write_csv(items_norm, out / "items.csv")
    write_csv(promos_norm, out / "promotions.csv")
    write_csv(stores_norm, out / "stores.csv")

if __name__ == "__main__":
    main()
