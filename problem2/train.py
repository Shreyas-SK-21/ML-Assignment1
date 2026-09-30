"""
train.py
--------
Training pipeline for Problem 2 (var2).

Exposes a single entry-point  `train_model()`  that:
  1. Expands features to the specified polynomial degree
  2. Fits using the Normal Equation
  3. Evaluates on both train and validation sets
  4. Sweeps over a range of degrees so you can compare them

Parameters you can change at the top:
  - DEGREE_MIN, DEGREE_MAX : range for the degree sweep
"""

import numpy as np
import model as M
import evaluate as E


# ─────────────────────────────────────────────
#  ★  CHANGE THESE AS NEEDED  ★
# ─────────────────────────────────────────────
DEGREE_MIN = 1
DEGREE_MAX = 12    # var2 can go up to degree 20 per assignment; sweep up to 12 first
# ─────────────────────────────────────────────




def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    degree: int = M.DEGREE,
    feature_names: list = None,
    verbose: bool = True,
):
    """
    Fit a polynomial regression model of the given degree.

    Parameters
    ----------
    X_train, y_train : training data
    X_val,   y_val   : validation data
    degree           : polynomial degree to use
    feature_names    : list of feature name strings (for display only)
    verbose          : whether to print metrics

    Returns
    -------
    w           : np.ndarray  — fitted weights
    powers      : list of exponent tuples
    train_metrics : dict with mse, rmse, r2, mae
    val_metrics   : dict with mse, rmse, r2, mae
    """
    # ── 1. Expand features ────────────────────────────────────────────────
    Phi_train, powers = M.expand_features(X_train, degree=degree)
    Phi_val,   _      = M.expand_features(X_val,   degree=degree)

    # ── 2. Fit (Normal Equation) ──────────────────────────────────────────
    w = M.fit(Phi_train, y_train)

    # ── 3. Predict ────────────────────────────────────────────────────────
    y_pred_train = M.predict(Phi_train, w)
    y_pred_val   = M.predict(Phi_val,   w)

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    if verbose:
        print(f"\n-- Degree {degree} " + "-"*30)
        print(f"  Num terms : {len(powers)}")
    train_metrics = E.evaluate_all(y_train, y_pred_train, label=f"Train (deg={degree})" if verbose else "")
    val_metrics   = E.evaluate_all(y_val,   y_pred_val,   label=f"Val   (deg={degree})" if verbose else "")

    if verbose and feature_names:
        labels = M.get_term_labels(feature_names, powers)
        print(f"\n  Top-5 weights by |magnitude|:")
        order = np.argsort(np.abs(w))[::-1][:5]
        for rank, i in enumerate(order, 1):
            print(f"    {rank}. {labels[i]:20s}  w = {w[i]:+.4f}")

    return w, powers, train_metrics, val_metrics


def sweep_degrees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    degree_min: int = DEGREE_MIN,
    degree_max: int = DEGREE_MAX,
    feature_names: list = None,
):
    """
    Train models across a range of polynomial degrees and collect metrics.
    Used to generate the degree-vs-metric plots in the notebook.

    Returns
    -------
    results : list of dicts, one per degree, each containing:
              { degree, train_mse, val_mse, train_r2, val_r2,
                train_rmse, val_rmse, train_mae, val_mae }
    """
    results = []
    for d in range(degree_min, degree_max + 1):
        _, _, tm, vm = train_model(
            X_train, y_train, X_val, y_val,
            degree=d, feature_names=feature_names, verbose=False
        )
        results.append({
            "degree"     : d,
            "train_mse"  : tm["mse"],  "val_mse"  : vm["mse"],
            "train_rmse" : tm["rmse"], "val_rmse" : vm["rmse"],
            "train_r2"   : tm["r2"],   "val_r2"   : vm["r2"],
            "train_mae"  : tm["mae"],  "val_mae"  : vm["mae"],
        })
    return results
