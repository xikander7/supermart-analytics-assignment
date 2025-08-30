# Headless runner: loads data, builds features, trains model, writes report/quick_insights.md
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.io import load_cleaned_master, detect_datetime, add_time_features, build_promo_flag
from src.features import build_feature_matrix
from src.modeling import train_evaluate
from src.insights import promotion_uplift, top_groups, write_quick_insights

def main():
    df = load_cleaned_master()
    df = detect_datetime(df)
    df = add_time_features(df)
    df = build_promo_flag(df)

    X, y, num_cols, cat_cols = build_feature_matrix(df, target_col="amount")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, metrics = train_evaluate(X_train, X_test, y_train, y_test, num_cols, cat_cols)

    uplift = promotion_uplift(df)
    prov  = top_groups(df, "province", n=5)
    stores= top_groups(df, "supermarket_no", n=5)
    items = top_groups(df, "code", n=5)

    path = write_quick_insights("report", metrics, uplift, [
        ("Top provinces (by revenue)", prov),
        ("Top stores (by revenue)", stores),
        ("Top items (by revenue)", items),
    ])
    print("Wrote insights to:", path.resolve())
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
