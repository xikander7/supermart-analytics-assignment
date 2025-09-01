"""Model training utilities."""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib

LOGGER = logging.getLogger(__name__)

def train_regressor(df: pd.DataFrame, target: str, random_state: int = 42, rf_params: dict | None = None):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, shuffle=True
    )
    model = RandomForestRegressor(**(rf_params or {}))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    return model, metrics, (X_test, y_test, y_pred)

def save_model(model, path: str):
    joblib.dump(model, path)
    LOGGER.info("Saved model -> %s", path)
