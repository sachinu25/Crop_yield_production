"""
Data-quality rules and regression metrics for model evaluation.

The APY source data contains a small number of physically impossible yields in
the 2018-2020 test years (e.g. guar seed at 485 t/ha against a 17-year maximum
of 3.6 t/ha). Training data is clipped during preprocessing, but test data is
not, so these records dominate RMSE/R². The rule below flags them using
statistics from the TRAINING years only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

IMPOSSIBLE_YIELD_FACTOR = 2.0


def crop_max_yield(y_train, crops_train) -> pd.Series:
    """Highest yield observed per crop in the training years."""
    return pd.Series(np.asarray(y_train, dtype=float)).groupby(np.asarray(crops_train)).max()


def impossible_yield_mask(y, crops, train_max: pd.Series,
                          factor: float = IMPOSSIBLE_YIELD_FACTOR) -> np.ndarray:
    """True where yield exceeds `factor` x that crop's training maximum.

    Crops never seen in training are never flagged.
    """
    limit = pd.Series(np.asarray(crops)).map(train_max).to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    return np.where(np.isnan(limit), False, y > factor * limit)


def regression_metrics(y_true, y_pred) -> dict:
    """R², RMSE, MAE and median absolute percentage error (predictions clipped at 0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "median_ape_pct": round(float(np.median(np.abs(y_pred - y_true) / np.maximum(y_true, 1e-3)) * 100), 2),
        "n": int(len(y_true)),
    }
