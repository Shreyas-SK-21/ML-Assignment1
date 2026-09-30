"""
evaluate.py
-----------
Evaluation metrics computed from scratch using NumPy.

Metrics implemented:
  - MSE  : Mean Squared Error
  - RMSE : Root Mean Squared Error
  - R²   : Coefficient of Determination
  - MAE  : Mean Absolute Error
"""

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (Coefficient of Determination).
    R² = 1 - SS_res / SS_tot
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """
    Compute and print all metrics, return as dict.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    label  : str — prefix label for printing (e.g. "Train" / "Val")

    Returns
    -------
    dict with keys: mse, rmse, r2, mae
    """
    metrics = {
        "mse":  mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
    }
    tag = f"[{label}] " if label else ""
    print(f"{tag}MSE={metrics['mse']:.6f}  RMSE={metrics['rmse']:.6f}  "
          f"R²={metrics['r2']:.6f}  MAE={metrics['mae']:.6f}")
    return metrics
