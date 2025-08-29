#!/usr/bin/env python3
"""
Train a baseline supervised model to predict sales 'amount'.
- Reads features.parquet
- Splits train/test
- Trains LinearRegression (baseline) and saves model + metrics
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.io_utils import PROCESSED_DIR, write_df
from src.metrics import regression_report, pretty_print_report

ARTIFACTS = PROCESSED_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    target = "amount"
    numeric = [c for c in ["units","price","size","year","month","weekofyear","dow","hour"] if c in df.columns]
    categoricals = [c for c in ["type","brand","province","feature","display","supermarket_no","post_code"] if c in df.columns]

    df = df.dropna(subset=[target])

    X = df[numeric + categoricals].copy()
    y = df[target].values

    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categoricals),
    ])

    model = Pipeline([
        ("pre", pre),
        ("lr", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    rep = regression_report(y_test, y_hat)
    pretty_print_report("Linear(Baseline)", rep)

    joblib.dump(model, ARTIFACTS / "model.joblib")
    import json
    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print("✅ Saved model.joblib and metrics.json to data/processed/artifacts/")

if __name__ == "__main__":
    main()
