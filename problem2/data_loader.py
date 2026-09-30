"""
data_loader.py
--------------
Loads the var1 dataset and returns train/val/test splits.

All parameters are exposed so you can change them freely:
  - TRAIN_CSV    : path to training CSV
  - TEST_CSV     : path to test CSV
  - FEATURES     : list of feature column names to use
  - TARGET       : name of target column
  - VAL_RATIO    : fraction of training data held out for validation
  - RANDOM_SEED  : seed for reproducibility
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  ★  CHANGE THESE AS NEEDED  ★
# ─────────────────────────────────────────────
TRAIN_CSV   = "../../IMT2024045/IMT2024045_train_var2.csv"
TEST_CSV    = "../../IMT2024045/IMT2024045_test_var2.csv"
FEATURES    = ["x1"]      # Assignment says optimal = first feature
TARGET      = "y"
VAL_RATIO   = 0.20                     # 80/20 split
RANDOM_SEED = 42
# ─────────────────────────────────────────────


def load_data(
    train_csv: str = TRAIN_CSV,
    test_csv: str  = TEST_CSV,
    features: list = FEATURES,
    target: str    = TARGET,
    val_ratio: float   = VAL_RATIO,
    random_seed: int   = RANDOM_SEED,
):
    """
    Load raw CSVs and return numpy arrays for train, validation and test.

    Returns
    -------
    X_train : np.ndarray  shape (n_train, len(features))
    y_train : np.ndarray  shape (n_train,)
    X_val   : np.ndarray  shape (n_val,   len(features))
    y_val   : np.ndarray  shape (n_val,)
    X_test  : np.ndarray  shape (n_test,  len(features))
    feature_names : list[str]
    """
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # ── Sanity checks ──────────────────────────────────────────────────────
    missing_train = [f for f in features if f not in train_df.columns]
    missing_test  = [f for f in features if f not in test_df.columns]
    if missing_train:
        raise ValueError(f"Features {missing_train} not found in training CSV.")
    if missing_test:
        raise ValueError(f"Features {missing_test} not found in test CSV.")
    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not found in training CSV.")

    X_all = train_df[features].values.astype(float)
    y_all = train_df[target].values.astype(float)
    X_test = test_df[features].values.astype(float)

    # ── Train / Validation split ───────────────────────────────────────────
    rng = np.random.default_rng(random_seed)
    n   = len(X_all)
    idx = rng.permutation(n)

    n_val   = int(np.floor(val_ratio * n))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

    print(f"[data_loader] Training   : {X_train.shape[0]} samples")
    print(f"[data_loader] Validation : {X_val.shape[0]}   samples")
    print(f"[data_loader] Test       : {X_test.shape[0]}  samples")
    print(f"[data_loader] Features   : {features}")

    return X_train, y_train, X_val, y_val, X_test, features
