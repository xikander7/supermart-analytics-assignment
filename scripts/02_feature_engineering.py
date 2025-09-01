#!/usr/bin/env python
\"\"\"Feature engineering and aggregation.\"\"\"
import logging, sys
from pathlib import Path
import pandas as pd
from src.io import load_config
from src.features import add_time_features, safe_numeric, agg_daily_store_sales

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a")
    ],
)
LOGGER = logging.getLogger("02_feature_engineering")

def main():
    cfg = load_config("config.yaml")
    processed = Path(cfg["paths"]["data_processed"])

    sales = pd.read_parquet(processed / "sales.parquet")
    promos = pd.read_parquet(processed / "promotions.parquet")
    stores = pd.read_parquet(processed / "stores.parquet")

    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales = safe_numeric(sales, ["amount", "quantity", "price"])

    if set(["item_id", "store_id", "start_date", "end_date"]).issubset(promos.columns):
        promos["start_date"] = pd.to_datetime(promos["start_date"], errors="coerce")
        promos["end_date"] = pd.to_datetime(promos["end_date"], errors="coerce")
        base = sales[["date", "item_id", "store_id"]].copy()
        base = base.merge(promos[["item_id", "store_id", "start_date", "end_date", "promo_flag"]],
                          on=["item_id", "store_id"], how="left")
        mask = (base["date"] >= base["start_date"]) & (base["date"] <= base["end_date"])
        sales["promo_flag"] = mask.astype(int)
    elif "promo_flag" in promos.columns:
        sales = sales.merge(promos[["item_id", "promo_flag"]].drop_duplicates(), on="item_id", how="left")
        sales["promo_flag"] = sales["promo_flag"].fillna(0).astype(int)
    else:
        sales["promo_flag"] = 0

    daily = agg_daily_store_sales(sales, "date", "store_id", "amount")
    daily = add_time_features(daily, "date")

    for c in ["state", "city"]:
        if c in stores.columns:
            daily = daily.merge(stores[["store_id", c]].drop_duplicates(), on="store_id", how="left")

    cat_cols = [c for c in ["state", "city"] if c in daily.columns]
    daily_enc = pd.get_dummies(daily, columns=cat_cols, dummy_na=True)

    out_path = processed / "features.parquet"
    daily_enc.to_parquet(out_path, index=False)
    LOGGER.info("Features saved -> %s", out_path)

if __name__ == "__main__":
    main()
