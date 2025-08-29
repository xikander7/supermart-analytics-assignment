"""Join cleaned tables and build a feature matrix (features.csv) + label.csv."""
import json
import pandas as pd
from src.utils import PROC, ensure_dirs
from src.features import join_all, build_feature_matrix
from src.io import read_table

def main(target='units'):
    ensure_dirs()
    items = read_table('items')
    sales = read_table('sales')
    promo = read_table('promotion')
    stores = read_table('supermarkets')

    df = join_all(sales, items, promo, stores)
    X, y, cols = build_feature_matrix(df, target=target)

    (PROC / "features.csv").parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(PROC / "features.csv", index=False)
    y.to_frame("label").to_csv(PROC / "label.csv", index=False)
    (PROC / "feature_cols.json").write_text(json.dumps(cols, indent=2))

    print(f"✅ features.csv created with {X.shape[0]:,} rows and {X.shape[1]:,} columns. Target = {target}")

if __name__ == "__main__":
    main()
