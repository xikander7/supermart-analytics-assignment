import numpy as np

def build_feature_matrix(df, target_col="amount"):
    if target_col not in df.columns:
        raise ValueError(f"Expected '{target_col}' as target column.")
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col], errors="ignore").copy()

    # Remove raw datetime column from features if present
    if "transaction_time" in X.columns:
        X = X.drop(columns=["transaction_time"])

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Simple imputation
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    return X[num_cols + cat_cols], y, num_cols, cat_cols
