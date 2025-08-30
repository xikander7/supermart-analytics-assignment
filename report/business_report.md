# Supermart Assignment — Business Report
_Generated: 2025-08-30_

## 1) Task Overview
Two parts: (a) use supervised ML to generate **business value** from supermarket transactions; (b) optional **Maze RL**. I completed both in a simple, reproducible setup.

## 2) Data Cleaning & Transformation
- Unified timestamp as **`transaction_time`** when any date-like column exists (e.g., `time_of_transactions`, `date`, `timestamp`).  
- Normalized promo flags from `feature`/`display` → **`promo_flag` ∈ {0,1}**.  
- Ensured numeric types for `amount`/`units`; median imputation for numeric NAs; safe string fill for categoricals.  
- Final analysis table: **`data/cleaned_master.csv`**.

## 3) Supervised Learning (Problem & Model)
**Problem.** Predict **`amount`** per transaction and turn the model + features into **actionable insights** (promotions, store mix, assortment).  
**Model.** `Ridge` regression in a `Pipeline` with:  
- `StandardScaler` on numeric features; `OneHotEncoder` on categoricals.  
- Time features: hour/day-of-week/month when timestamp is available.  
**Metrics.** MAE / RMSE / R² printed by `notebooks/02_model_and_insights.ipynb`.

## 4) Business Insights (at least two)
1. **Promotion Uplift** — difference in average `amount` between promo vs non‑promo transactions (via `promo_flag`).  
2. **Top Revenue Contributors** — top **provinces / stores / items** by revenue.  
3. *(Optional extra)* Seasonality indicators (month/DOW) if timestamp exists.

The notebook writes a quick summary to `report/quick_insights.md` so you can paste directly into slides/email.

## 5) Maze RL (Optional)
`maze/gridworld_qlearning.py` implements a toy GridWorld with **Q-learning**. Train via:
```bash
python maze/gridworld_qlearning.py
```
It learns a greedy policy to reach the goal while avoiding walls.

## 6) How to Run
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```
In Jupyter:
1. **01_prepare_data.ipynb** — load & sanity‑check.  
2. **02_model_and_insights.ipynb** — train, evaluate, and write insights.

## 7) Notes
- Structure is intentionally minimal: **data/**, **notebooks/**, **report/**, **maze/**.  
- Everything is defensive: code runs even if some optional columns are missing.
