import pandas as pd
import numpy as np

def standardize_cols(df):
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def parse_time_columns(sales):
    # 'time' may appear under different names
    col = None
    for cand in ['time', 'time_of_transactions', 'transaction_time']:
        if cand in sales.columns:
            col = cand
            break
    if col:
        sales[col] = pd.to_datetime(sales[col], errors='coerce')
        sales['year'] = sales[col].dt.year
        sales['month'] = sales[col].dt.month
        sales['dayofweek'] = sales[col].dt.dayofweek
        sales['weekofyear'] = sales[col].dt.isocalendar().week.astype(int)
        sales['is_weekend'] = sales['dayofweek'].isin([5,6]).astype(int)
    else:
        # Fallback synthetic time
        n = len(sales)
        base = pd.Timestamp('2024-01-01')
        syn = [base + pd.Timedelta(days=int(i%365)) for i in range(n)]
        sales['synthetic_time'] = syn
        sales['year'] = [t.year for t in syn]
        sales['month'] = [t.month for t in syn]
        sales['dayofweek'] = [t.dayofweek for t in syn]
        sales['weekofyear'] = [int(pd.Timestamp(t).isocalendar().week) for t in syn]
        sales['is_weekend'] = (sales['dayofweek'].isin([5,6])).astype(int)
    return sales

def join_all(sales, items, promo, stores):
    # Standardize columns
    sales = standardize_cols(sales)
    items = standardize_cols(items)
    promo = standardize_cols(promo)
    stores = standardize_cols(stores)

    # Normalize schema basics
    sales.rename(columns={'supermarket_no': 'supermarket_no', 'supermarket_number':'supermarket_no'}, inplace=True)
    promo.rename(columns={'supermarket_no': 'supermarket_no', 'supermarket_number':'supermarket_no'}, inplace=True)
    stores.rename(columns={'supermarket_no': 'supermarket_no', 'supermarket_number':'supermarket_no'}, inplace=True)

    # Time features on sales
    sales = parse_time_columns(sales)

    # Align week keys
    if 'week' in promo.columns:
        promo['weekofyear'] = promo['week']
    if 'weekofyear' not in sales.columns and 'week' in sales.columns:
        sales.rename(columns={'week': 'weekofyear'}, inplace=True)

    # Join items
    df = sales.merge(items, on='code', how='left', suffixes=('', '_item'))
    # Join promo by (code, store[, weekofyear])
    on_left = ['code', 'supermarket_no']
    on_right = ['code', 'supermarket_no']
    if 'weekofyear' in sales.columns and 'weekofyear' in promo.columns:
        on_left.append('weekofyear')
        on_right.append('weekofyear')
    df = df.merge(promo, left_on=on_left, right_on=on_right, how='left', suffixes=('', '_promo'))
    # Join stores
    df = df.merge(stores, on='supermarket_no', how='left', suffixes=('', '_store'))

    # Encode promo flags
    for col in ['feature','display']:
        if col in df.columns:
            s = df[col].fillna(0)
            if s.dtype == 'O':
                s = s.astype(str).str.strip().str.lower().map({'y':1,'yes':1,'1':1,'true':1,'t':1})
                s = s.fillna(0)
            df[col] = s.astype(int)
    return df

def build_feature_matrix(df, target='units'):
    # Encode common categoricals
    for c in ['type','brand','province','post-code']:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    # Derived promos
    if 'feature' in df.columns and 'display' in df.columns:
        df['promo_any'] = ((df['feature']>0) | (df['display']>0)).astype(int)
        df['promo_combo'] = (df['feature'] * 2 + df['display']).astype(int)

    # Target fallback
    if target not in df.columns:
        if target == 'units' and 'amount' in df.columns:
            df['units'] = (df['amount'] / max(df['amount'].mean(), 1)).clip(lower=0)
        else:
            df[target] = 0.0

    # Select numeric, non-datetime features
    drop_cols = {'amount','units','description','basket','voucher','customerid','synthetic_time','time'}
    feature_cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        dt = df[c].dtype
        if getattr(dt, 'kind', None) == 'M':
            continue  # datetime
        if df[c].dtype == 'O':
            continue  # object/string
        feature_cols.append(c)

    X = df[feature_cols].fillna(0)
    y = df[target].astype(float)
    return X, y, feature_cols
