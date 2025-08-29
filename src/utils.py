import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW = BASE_DIR / "data" / "raw"
PROC = BASE_DIR / "data" / "processed"
REPORT = BASE_DIR / "report"
FIG = REPORT / "figures"

def ensure_dirs():
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
