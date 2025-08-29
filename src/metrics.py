from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_report(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def pretty_print_report(name: str, report: dict):
    print(f"{name:>12} | MAE: {report['MAE']:.3f} | RMSE: {report['RMSE']:.3f} | R²: {report['R2']:.3f}")
