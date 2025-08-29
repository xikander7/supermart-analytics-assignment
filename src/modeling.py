import json
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from .metrics import evaluate_regression

def train_random_forest(X, y, random_state=42, n_estimators=200, max_depth=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    metrics = evaluate_regression(model, X_test, y_test)
    return model, metrics

def save_model_and_metrics(model, metrics, model_path, metrics_path):
    dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
