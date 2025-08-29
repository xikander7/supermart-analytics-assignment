"""Aggregate business insights + save charts and a compact PDF report."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from src.utils import PROC, REPORT, FIG, ensure_dirs
from src.plots import barh, line
from src.io import read_table

def promo_lift_estimate(sales_df, promo_df):
    sales_df = sales_df.copy()
    # Ensure lowercase and time parsed
    if 'time' in sales_df.columns:
        sales_df['time'] = pd.to_datetime(sales_df['time'], errors='coerce')
        sales_df['weekofyear'] = sales_df['time'].dt.isocalendar().week.astype('Int64')
    else:
        return np.nan, pd.DataFrame()

    if 'week' in promo_df.columns:
        tmp = promo_df.rename(columns={'week':'weekofyear'})
        m = sales_df.merge(tmp, on=['code','supermarket_no','weekofyear'], how='left')
    else:
        m = sales_df.merge(promo_df, on=['code','supermarket_no'], how='left')

    for c in ['feature','display']:
        if c in m.columns:
            m[c] = (m[c].fillna(0) > 0).astype(int)
    m['promo_any'] = ((m.get('feature',0)>0) | (m.get('display',0)>0)).astype(int)
    grp = m.groupby('promo_any')['units'].mean().rename('avg_units').reset_index()
    try:
        base = float(grp.loc[grp['promo_any']==0, 'avg_units'].values[0])
        promo = float(grp.loc[grp['promo_any']==1, 'avg_units'].values[0])
        lift = (promo - base) / max(base, 1e-6)
    except Exception:
        lift = np.nan
    return lift, grp

def main():
    ensure_dirs()
    items = read_table('items')
    sales = read_table('sales')
    promo = read_table('promotion')
    stores = read_table('supermarkets')

    # Basic KPIs
    total_sales = float(sales['amount'].sum()) if 'amount' in sales.columns else float('nan')
    total_units = float(sales['units'].sum()) if 'units' in sales.columns else float('nan')

    # Promo lift
    plift, grp = promo_lift_estimate(sales, promo)
    insights = {}
    if isinstance(plift, float) and not np.isnan(plift):
        insights['promo_lift_pct'] = round(plift*100, 2)

    # Weekly units
    if 'time' in sales.columns:
        s = sales.copy()
        s['time'] = pd.to_datetime(s['time'], errors='coerce')
        weekly = s.groupby(s['time'].dt.isocalendar().week)['units'].sum().reset_index()
        line(weekly['week'], weekly['units'], "Weekly Units (All Stores)", "ISO Week", "Units", FIG / "weekly_units.png")

    # Top stores by units
    if 'supermarket_no' in sales.columns:
        tops = sales.groupby('supermarket_no')['units'].sum().sort_values(ascending=False).head(10)
        barh(list(tops.index[::-1]), list(tops.values[::-1]), "Top 10 Stores by Units", "Total Units", FIG / "top_stores.png")

    if insights:
        (REPORT / "insights.json").write_text(json.dumps(insights, indent=2))

    # PDF report
    with PdfPages(REPORT / "report.pdf") as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.3, 11.7))
        ax.axis('off')
        txt = """Supermart Analytics — Model & Insights

        • Objective: Clean data, build supervised model (predict Units), and extract business insights.
        • Datasets: Items, Sales, Promotion, Supermarkets.
        • Outputs: Trained model, metrics, figures, and this PDF.

        Highlights
        ----------
        - End-to-end pipeline with synthetic fallback when CSVs are missing.
        - RandomForestRegressor baseline with feature importances.
        - Business insight: estimated promotion lift on unit sales.
        - Top performing stores (by units).
        - Bonus: RL maze demo (see PNG in figures)."""
        ax.text(0.05, 0.95, txt, va='top', wrap=True, fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        # Metrics + Feature importances
        fig, ax = plt.subplots(figsize=(8.3, 11.7))
        ax.axis('off')
        metrics_path = PROC / "metrics.json"
        metrics_text = "No metrics found."
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            metrics_text = json.dumps(metrics, indent=2)
        ax.text(0.05, 0.95, "Model Metrics", va='top', fontsize=14, weight='bold')
        ax.text(0.05, 0.90, metrics_text, va='top', family='monospace', fontsize=10)
        # attach image
        imp = (REPORT / "figures" / "feature_importances.png")
        if imp.exists():
            img = plt.imread(imp)
            ax.imshow(img, extent=(0.05, 0.95, 0.05, 0.55), aspect='auto')
        pdf.savefig(fig); plt.close(fig)

        # Weekly units + Top stores
        for name in ["weekly_units.png", "top_stores.png", "maze_policy.png"]:
            p = (REPORT / "figures" / name)
            if p.exists():
                fig, ax = plt.subplots(figsize=(8.3, 11.7))
                ax.axis('off')
                img = plt.imread(p)
                ax.imshow(img)
                pdf.savefig(fig); plt.close(fig)

    print("🧾  Wrote report/report.pdf and figures.")

if __name__ == "__main__":
    main()
