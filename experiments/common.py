"""Shared loading + metrics for crop-yield experiments (uses the repo's saved checkpoints)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA = Path(__file__).resolve().parents[1] / "checkpoints"  # repo checkpoints (train 2001-2017, test 2018-2020)
CAT_COLS = ["season", "crop", "state_name", "crop_season", "state_crop",
            "geo_cluster_10", "geo_cluster_30", "geo_cluster_50"]


def load():
    Xtr = pd.read_parquet(DATA / "X_train_cat.parquet")
    Xte = pd.read_parquet(DATA / "X_test_cat.parquet")
    ytr = np.expm1(np.load(DATA / "y_train.npy"))  # stored as log1p(yield)
    yte = np.load(DATA / "y_test.npy")             # stored raw
    return Xtr, ytr, Xte, yte


def to_lgb(X, cats_ref=None):
    """String categoricals -> pandas category with categories fixed from the training frame."""
    X = X.copy()
    for c in CAT_COLS:
        cats = cats_ref[c] if cats_ref is not None else sorted(X[c].astype(str).unique())
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return X


def cat_levels(X):
    return {c: sorted(X[c].astype(str).unique()) for c in CAT_COLS}


def baseline(X):
    """Naive forecast: last-3-year mean yield of the same district+crop (falls back to crop-level)."""
    b = X["roll3_mean_district_crop"].to_numpy(dtype=float).copy()
    fallback = X["roll3_mean_crop"].to_numpy(dtype=float)
    bad = ~np.isfinite(b) | (b <= 0)
    b[bad] = fallback[bad]
    return np.clip(b, 1e-3, None)


def metrics(y, p, crops=None):
    p = np.clip(p, 0, None)
    out = {
        "R2": r2_score(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": mean_absolute_error(y, p),
        "R2_log": r2_score(np.log1p(y), np.log1p(p)),
        "MedAPE%": float(np.median(np.abs(p - y) / np.maximum(y, 1e-3)) * 100),
    }
    if crops is not None:
        df = pd.DataFrame({"c": np.asarray(crops), "y": y, "p": p})
        per = [r2_score(g.y, g.p) for _, g in df.groupby("c") if len(g) >= 30 and g.y.var() > 0]
        out["median_per_crop_R2"] = float(np.median(per))
    return {k: round(float(v), 4) for k, v in out.items()}
