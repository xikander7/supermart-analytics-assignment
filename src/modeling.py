from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ])
    model = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])
    return model

def train_evaluate(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    model = build_pipeline(num_cols, cat_cols)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    return model, {"MAE": mae, "RMSE": rmse, "R2": r2}
