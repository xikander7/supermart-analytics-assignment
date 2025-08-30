from pathlib import Path
import pandas as pd
import numpy as np

def promotion_uplift(df):
    if "promo_flag" not in df.columns or "amount" not in df.columns:
        return np.nan
    base = df.loc[df["promo_flag"]==0, "amount"].mean() if (df["promo_flag"]==0).any() else np.nan
    promo = df.loc[df["promo_flag"]==1, "amount"].mean() if (df["promo_flag"]==1).any() else np.nan
    if pd.notna(base) and base>0 and pd.notna(promo):
        return (promo - base)/base*100
    return np.nan

def top_groups(df, by_col, n=5):
    if by_col not in df.columns or "amount" not in df.columns:
        return None
    return (df.groupby(by_col, as_index=False)["amount"]
              .sum()
              .sort_values("amount", ascending=False)
              .head(n))

def write_quick_insights(report_dir, metrics, uplift, tables):
    report_dir = Path(report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    if metrics:
        lines.append(f"**Model performance** — MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.3f}")
    lines.append(f"Promotion uplift: {'N/A' if pd.isna(uplift) else f'{uplift:.2f}%'}")
    for title, tbl in tables:
        if tbl is not None:
            lines.append(f"\n{title}")
            for _, r in tbl.iterrows():
                # key = first col name
                k = tbl.columns[0]
                lines.append(f"- {r[k]}: {r['amount']:.0f}")
    (report_dir / "quick_insights.md").write_text("\n".join(lines), encoding="utf-8")
    return report_dir / "quick_insights.md"
