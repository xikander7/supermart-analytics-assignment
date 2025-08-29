"""Train a simple RandomForest on features.parquet and store artifacts + metrics."""
import json
import pandas as pd
from joblib import dump
from pathlib import Path
from src.utils import PROC, ensure_dirs, REPORT, FIG
from src.modeling import train_random_forest, save_model_and_metrics

def main():
    ensure_dirs()
    X = pd.read_csv(PROC / "features.csv")
    y = pd.read_csv(PROC / "label.csv")["label"]
    model, metrics = train_random_forest(X, y)
    save_model_and_metrics(model, metrics, PROC / "model.joblib", PROC / "metrics.json")
    print("✅ Model trained. Metrics:", metrics)

    # Feature importance plot
    import numpy as np
    from src.plots import barh
    cols = json.loads((PROC / "feature_cols.json").read_text())
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        idx = np.argsort(importances)[::-1][:15]
        names = [cols[i] for i in idx]
        vals = [float(importances[i]) for i in idx]
        barh(names, vals, "Top Feature Importances", "Importance", FIG / "feature_importances.png")
        print("🖼  Saved feature_importances.png")

if __name__ == "__main__":
    main()
