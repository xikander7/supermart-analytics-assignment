#!/usr/bin/env python3
"""
Generate a couple of business insights and figures.
Examples:
- Uplift during promotions (feature/display) by item type
- Top performing supermarkets by normalized sales
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.io_utils import PROCESSED_DIR

FIG_DIR = Path("report/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROCESSED_DIR / "insights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_and_save(df, title, filename, x, y):
    ax = df.plot(kind="bar", x=x, y=y, legend=False, figsize=(8,4))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()

def main():
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    # 1) Promo uplift by item type
    if {"feature","display","type","amount"}.issubset(features.columns):
        tmp = features.copy()
        tmp["is_promo"] = tmp["feature"].notna() | tmp["display"].notna()
        agg = tmp.groupby(["type","is_promo"], dropna=False)["amount"].mean().reset_index()
        agg.to_csv(OUT_DIR / "promo_uplift_by_type.csv", index=False)
        pvt = agg.pivot(index="type", columns="is_promo", values="amount").fillna(0).reset_index()
        if True in pvt.columns and False in pvt.columns:
            pvt["uplift_vs_no_promo"] = pvt[True] - pvt[False]
            plot_and_save(pvt, "Promo Uplift by Type (avg amount)", "promo_uplift_by_type.png", "type", "uplift_vs_no_promo")

    # 2) Top supermarkets by normalized sales
    if {"supermarket_no","amount"}.issubset(features.columns):
        store = features.groupby("supermarket_no")["amount"].sum().reset_index()
        store["z"] = (store["amount"] - store["amount"].mean()) / store["amount"].std(ddof=0)
        top = store.sort_values("z", ascending=False).head(10)
        top.to_csv(OUT_DIR / "top_supermarkets.csv", index=False)
        plot_and_save(top, "Top 10 Supermarkets (z-score of total sales)", "top_supermarkets.png", "supermarket_no", "z")

    print("✅ Insights saved under report/figures and data/processed/insights")

if __name__ == "__main__":
    main()
