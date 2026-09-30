"""
model.py
--------
Polynomial feature expansion and the regression model.

Everything is NumPy-only (no sklearn).

The polynomial expansion uses the multivariate convention:
  degree d, p features  →  all monomials x1^a1 * x2^a2 * ... * xp^ap
  where a1 + a2 + ... + ap <= d

Parameters you can change:
  - DEGREE : polynomial degree (must be >= 1)
"""

import numpy as np
from itertools import combinations_with_replacement


# ─────────────────────────────────────────────
#  ★  CHANGE THIS AS NEEDED  ★
# ─────────────────────────────────────────────
DEGREE = 4     # Assignment says optimal degree = 4 for var2
# ─────────────────────────────────────────────


def _get_poly_powers(n_features: int, degree: int) -> list[tuple]:
    """
    Return list of exponent tuples for all monomials up to the given degree.
    E.g. for n_features=2, degree=2:
      [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
      i.e. [1, x1, x2, x1^2, x1*x2, x2^2]
    """
    powers = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            exponents = [0] * n_features
            for idx in combo:
                exponents[idx] += 1
            powers.append(tuple(exponents))
    return powers


def expand_features(X: np.ndarray, degree: int = DEGREE) -> np.ndarray:
    """
    Transform raw feature matrix X (shape [n, p]) into polynomial feature
    matrix Phi (shape [n, num_terms]) where num_terms includes the bias term
    (the constant 1 monomial).

    Parameters
    ----------
    X      : np.ndarray of shape (n_samples, n_features)
    degree : int, polynomial degree

    Returns
    -------
    Phi    : np.ndarray of shape (n_samples, num_terms)
    powers : list of exponent tuples (for human-readable term labels)
    """
    n, p   = X.shape
    powers = _get_poly_powers(p, degree)

    Phi = np.ones((n, len(powers)), dtype=float)
    for col_idx, exps in enumerate(powers):
        for feat_idx, exp in enumerate(exps):
            if exp > 0:
                Phi[:, col_idx] *= X[:, feat_idx] ** exp

    return Phi, powers


def get_term_labels(feature_names: list, powers: list) -> list[str]:
    """
    Convert exponent tuples to human-readable strings.
    E.g. (2, 1, 0) with feature_names ['x1','x2','x3'] → 'x1^2·x2'
    """
    labels = []
    for exps in powers:
        parts = []
        for fname, e in zip(feature_names, exps):
            if e == 1:
                parts.append(fname)
            elif e > 1:
                parts.append(f"{fname}^{e}")
        labels.append("·".join(parts) if parts else "1")
    return labels


def fit(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit polynomial regression using the Normal Equation (closed-form):
        w = (Phi^T Phi)^{-1} Phi^T y

    Uses np.linalg.lstsq for numerical stability (handles near-singular cases).

    Parameters
    ----------
    Phi : np.ndarray of shape (n_samples, num_terms)
    y   : np.ndarray of shape (n_samples,)

    Returns
    -------
    w   : np.ndarray of shape (num_terms,)  — learned weights
    """
    w, residuals, rank, sv = np.linalg.lstsq(Phi, y, rcond=None)
    return w


def predict(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Generate predictions given expanded feature matrix and weights.

    Parameters
    ----------
    Phi : np.ndarray of shape (n_samples, num_terms)
    w   : np.ndarray of shape (num_terms,)

    Returns
    -------
    y_hat : np.ndarray of shape (n_samples,)
    """
    return Phi @ w
