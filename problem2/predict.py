"""
predict.py
----------
Inference module — generate predictions on the test set and save to CSV.

Parameters you can change:
  - OUTPUT_CSV : path/name of the output prediction file
"""

import numpy as np
import pandas as pd
import model as M


# ─────────────────────────────────────────────
#  ★  CHANGE THIS AS NEEDED  ★
# ─────────────────────────────────────────────
OUTPUT_CSV = "../../IMT2024045_pred_var2.csv"
# ─────────────────────────────────────────────


def generate_predictions(
    X_test: np.ndarray,
    w: np.ndarray,
    degree: int = M.DEGREE,
    output_csv: str = OUTPUT_CSV,
    save: bool = True,
) -> np.ndarray:
    """
    Expand test features and generate predictions using fitted weights.

    Parameters
    ----------
    X_test     : np.ndarray of shape (n_test, n_features)
    w          : np.ndarray — fitted weight vector
    degree     : int — must match the degree used during training
    output_csv : str — path to write the output CSV
    save       : bool — whether to write the CSV to disk

    Returns
    -------
    y_pred : np.ndarray of shape (n_test,)
    """
    Phi_test, _ = M.expand_features(X_test, degree=degree)
    y_pred = M.predict(Phi_test, w)

    if save:
        df_out = pd.DataFrame({"y": y_pred})
        df_out.to_csv(output_csv, index=False)
        print(f"[predict] Saved {len(y_pred)} predictions -> {output_csv}")

    print(f"[predict] y_pred  min={y_pred.min():.4f}  max={y_pred.max():.4f}  "
          f"mean={y_pred.mean():.4f}")

    return y_pred
