from pathlib import Path
import pandas as pd

def find_cleaned_master():
    candidates = [
        Path("data/cleaned_master.csv"),
        Path("data/processed/cleaned_master.csv"),
        Path("../data/cleaned_master.csv"),
        Path("../data/processed/cleaned_master.csv"),
    ]
    return next((p for p in candidates if p.exists()), None)

def load_cleaned_master():
    src = find_cleaned_master()
    if src is None:
        raise FileNotFoundError("cleaned_master.csv not found in data/ or data/processed/.")
    return pd.read_csv(src, low_memory=False)

def detect_datetime(df):
    candidates = ["transaction_time","time_of_transactions","transaction_date","date","datetime","timestamp","time"]
    src = next((c for c in candidates if c in df.columns), None)
    if src:
        df = df.copy()
        df["transaction_time"] = pd.to_datetime(df[src], errors="coerce")
    return df

def add_time_features(df):
    df = df.copy()
    if "transaction_time" in df.columns:
        df["hour"]  = df["transaction_time"].dt.hour
        df["dow"]   = df["transaction_time"].dt.dayofweek
        df["month"] = df["transaction_time"].dt.month
    return df

def build_promo_flag(df):
    df = df.copy()
    df["promo_flag"] = 0
    for c in ["feature","display"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan":"0","None":"0"})
            df.loc[df[c].ne("0"), "promo_flag"] = 1
    return df
