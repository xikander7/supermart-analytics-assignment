#!/usr/bin/env python
\"\"\"Train supervised model and generate business insights & plots.\"\"\"
import logging, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from src.io import load_config
from src.modeling import save_model
from src.metrics import promotion_uplift

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a")
    ],
)
LOGGER = logging.getLogger("03_train_model")

def main():
    cfg = load_config("config.yaml")
    processed = Path(cfg["paths"]["data_processed"])
    figures = Path(cfg["paths"]["figures"])
    models_dir = Path(cfg["paths"].get("models", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    feat = pd.read_parquet(processed / "features.parquet")
    target = cfg["ml"]["target"]
    drop_cols = [c for c in ["date", "year_week"] if c in feat.columns]
    Xy = feat.drop(columns=drop_cols).copy()
    if target not in Xy.columns:
        raise RuntimeError(f"Target {target} not found in features")
    y = Xy[target]
    X = Xy.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["ml"]["test_size"], random_state=cfg["ml"]["random_state"], shuffle=True
    )
    model = RandomForestRegressor(**cfg["ml"]["rf_params"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    (Path("report") / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    save_model(model, models_dir / "random_forest_sales.joblib")

    plt.figure()
    plt.scatter(y_test, y_pred, s=10)
    plt.xlabel("Actual daily_store_sales")
    plt.ylabel("Predicted daily_store_sales")
    plt.title("Predicted vs Actual (Test)")
    plt.tight_layout()
    plt.savefig(figures / "pred_vs_actual.png", dpi=150)
    plt.close()

    # Business insight 1: Promotion uplift
    try:
        sales = pd.read_parquet(processed / "sales.parquet")
        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
        if "promo_flag" not in sales.columns:
            sales["promo_flag"] = 0
        daily = sales.groupby([\"store_id\", pd.Grouper(key=\"date\", freq=\"D\")])[\"amount\"].sum().reset_index()
        daily = daily.rename(columns={\"amount\":\"daily_store_sales\"})
        promo_daily = sales.groupby([pd.Grouper(key=\"date\", freq=\"D\")])[\"promo_flag\"].max().reset_index()
        promo_daily = promo_daily.rename(columns={\"promo_flag\":\"promo_flag_daily\"})
        merged = daily.merge(promo_daily, on=\"date\", how=\"left\")
        uplift = promotion_uplift(merged, \"promo_flag_daily\", \"daily_store_sales\")
    except Exception as e:
        uplift = float(\"nan\")

    with open(\"report/promo_uplift.txt\", \"w\") as f:
        if np.isnan(uplift):
            f.write(\"Promotion uplift could not be computed with available data.\\n\")
        else:
            f.write(f\"Estimated promotion uplift (relative): {uplift:.4f}\\n\")

    try:
        importances = model.feature_importances_
        names = X.columns
        order = np.argsort(importances)[::-1][:20]
        plt.figure()
        plt.bar(range(len(order)), importances[order])
        plt.xticks(range(len(order)), [names[i] for i in order], rotation=60, ha=\"right\")
        plt.ylabel(\"Importance\")
        plt.title(\"Top Feature Importances (Random Forest)\")
        plt.tight_layout()
        plt.savefig(figures / \"feature_importance.png\", dpi=150)
        plt.close()
    except Exception:
        pass

    try:
        if \"date\" in feat.columns:
            ts = feat.groupby(\"date\")[target].sum().reset_index()
            plt.figure()
            plt.plot(ts[\"date\"], ts[target])
            plt.title(\"Total Daily Sales Over Time\")
            plt.xlabel(\"Date\")
            plt.ylabel(\"Sales\")
            plt.tight_layout()
            plt.savefig(figures / \"sales_over_time.png\", dpi=150)
            plt.close()
    except Exception:
        pass

    LOGGER.info(\"Metrics: %s\", metrics)
    print(json.dumps(metrics, indent=2))

if __name__ == \"__main__\":
    main()
