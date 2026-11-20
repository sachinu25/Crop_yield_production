"""
Train the production LightGBM crop-yield model and benchmark it out-of-time.

Pipeline
  1. Load the engineered feature checkpoints (60 leak-free features; train 2001-2017, test 2018-2020).
  2. Fit the crop-specific quality thresholds on TRAINING years only (2x the crop's max training yield).
  3. Tune LightGBM with Optuna on a dev split (train 2001-2014, validate 2015-2017). The test years are never used for tuning.
  4. Refit on 2001-2017, add a per-crop log-retransformation (smearing) correction.
  5. Benchmark naive baseline, CatBoost, XGBoost and LightGBM on the 2018-2020 test years,
     before and after the quality filter.
  6. Save artifacts under models/ for the Streamlit app:
       lightgbm_model.txt, lightgbm_artifacts.json (feature order, category levels, smearing factors),
       quality_thresholds.json, evaluation.json

Usage:  python scripts/train_lightgbm.py [--trials 40] [--skip-xgboost]
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb  # import before pandas (avoids a DLL load-order crash on some Windows setups)
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.data_quality import (  # noqa: E402
    IMPOSSIBLE_YIELD_FACTOR, crop_max_yield, impossible_yield_mask, regression_metrics,
)

CK, MODELS, REPORTS = ROOT / "checkpoints", ROOT / "models", ROOT / "reports"
CAT_COLS = ["season", "crop", "state_name", "crop_season", "state_crop",
            "geo_cluster_10", "geo_cluster_30", "geo_cluster_50"]
SEED = 42


def to_category(X: pd.DataFrame, levels: dict) -> pd.DataFrame:
    X = X.copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=levels[c])
    return X


def naive_baseline(X: pd.DataFrame) -> np.ndarray:
    """Last-3-year mean yield of the same district+crop (crop-level fallback)."""
    b = X["roll3_mean_district_crop"].to_numpy(dtype=float).copy()
    bad = ~np.isfinite(b) | (b <= 0)
    b[bad] = X["roll3_mean_crop"].to_numpy(dtype=float)[bad]
    return np.clip(b, 1e-3, None)


def smearing_factors(model, X, y_raw, crops) -> pd.Series:
    """Duan smearing per crop: corrects the downward bias of expm1(log-prediction)."""
    resid = np.log1p(y_raw) - model.predict(X)
    return pd.Series(np.exp(resid)).groupby(np.asarray(crops)).mean()


def apply_smearing(log_pred, crops, factors: pd.Series) -> np.ndarray:
    f = pd.Series(np.asarray(crops)).map(factors).fillna(1.0).to_numpy()
    return (np.expm1(log_pred) + 1.0) * f - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=1200, help="Optuna time budget in seconds")
    ap.add_argument("--skip-xgboost", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    # 1. data -----------------------------------------------------------------
    X_tr = pd.read_parquet(CK / "X_train_cat.parquet")
    X_te = pd.read_parquet(CK / "X_test_cat.parquet")
    y_tr = np.expm1(np.load(CK / "y_train.npy"))     # stored as log1p(yield)
    y_te = np.load(CK / "y_test.npy")                # stored raw
    features = list(X_tr.columns)
    levels = {c: sorted(X_tr[c].astype(str).unique()) for c in CAT_COLS}
    L_tr, L_te = to_category(X_tr, levels), to_category(X_te, levels)
    print(f"train {len(X_tr):,} rows ({X_tr.crop_year.min()}-{X_tr.crop_year.max()}), "
          f"test {len(X_te):,} rows ({X_te.crop_year.min()}-{X_te.crop_year.max()}), {len(features)} features")

    # 2. quality thresholds (training years only) ------------------------------
    crop_max = crop_max_yield(y_tr, X_tr["crop"])
    thresholds = (crop_max * IMPOSSIBLE_YIELD_FACTOR).round(4)
    flagged = impossible_yield_mask(y_te, X_te["crop"], crop_max)
    keep = ~flagged

    # 3. Optuna on the dev split -------------------------------------------------
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev_tr = (X_tr["crop_year"] <= 2014).to_numpy()
    dev_va = ~dev_tr
    ds_params = {"feature_pre_filter": False}  # lets Optuna vary min_child_samples on a shared Dataset
    d_tr = lgb.Dataset(L_tr[dev_tr], np.log1p(y_tr[dev_tr]), params=ds_params, free_raw_data=False)
    d_va = lgb.Dataset(L_tr[dev_va], np.log1p(y_tr[dev_va]), reference=d_tr, params=ds_params, free_raw_data=False)
    base_params = dict(objective="regression", bagging_freq=1, max_cat_to_onehot=8,
                       verbose=-1, seed=SEED, n_jobs=-1)

    def objective(trial):
        p = {**base_params,
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
             "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
             "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
             "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
             "cat_smooth": trial.suggest_float("cat_smooth", 5, 200, log=True)}
        m = lgb.train(p, d_tr, 5000, valid_sets=[d_va], callbacks=[lgb.early_stopping(200, verbose=False)])
        sm = smearing_factors(m, L_tr[dev_tr], y_tr[dev_tr], X_tr["crop"][dev_tr])
        pv = apply_smearing(m.predict(L_tr[dev_va]), X_tr["crop"][dev_va], sm)
        trial.set_user_attr("iters", m.best_iteration)
        return regression_metrics(y_tr[dev_va], pv)["r2"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    best = {**base_params, **study.best_params}
    n_rounds = int(study.best_trial.user_attrs["iters"] * 1.15)
    print(f"optuna: {len(study.trials)} trials, best dev R2 {study.best_value:.4f}, rounds {n_rounds}")

    # 4. refit on 2001-2017 ------------------------------------------------------
    model = lgb.train(best, lgb.Dataset(L_tr, np.log1p(y_tr)), n_rounds)
    smear = smearing_factors(model, L_tr, y_tr, X_tr["crop"])
    p_lgb = apply_smearing(model.predict(L_te), X_te["crop"], smear)

    # 5. benchmarks --------------------------------------------------------------
    preds = {"Naive baseline": naive_baseline(X_te), "LightGBM (Optuna)": p_lgb}
    cb = pd.read_csv(REPORTS / "catboost_predictions.csv")
    if len(cb) == len(y_te) and np.allclose(cb["True_Yield"].to_numpy(), y_te):
        preds["CatBoost"] = cb["Predicted_Yield"].to_numpy()
    if not args.skip_xgboost:
        import xgboost as xgb
        xp = json.loads((MODELS / "best_params.json").read_text())["XGBoost"]
        xgb_model = xgb.XGBRegressor(**xp, tree_method="hist", enable_categorical=True,
                                     max_cat_to_onehot=8, random_state=SEED, n_jobs=-1)
        xgb_model.fit(L_tr, np.log1p(y_tr))
        preds["XGBoost"] = np.expm1(xgb_model.predict(L_te))

    order = ["Naive baseline", "CatBoost", "XGBoost", "LightGBM (Optuna)"]
    bench = {name: {"full": regression_metrics(y_te, preds[name]),
                    "quality_filtered": regression_metrics(y_te[keep], preds[name][keep])}
             for name in order if name in preds}
    sq = (y_te - np.clip(p_lgb, 0, None)) ** 2
    worst = (pd.DataFrame({"crop": X_te["crop"].to_numpy(), "year": X_te["crop_year"].to_numpy(),
                           "actual": y_te.round(2), "predicted": p_lgb.round(2),
                           "train_max": X_te["crop"].map(crop_max).to_numpy().round(2), "sq": sq})
             .sort_values("sq", ascending=False).head(10).drop(columns="sq"))
    base_mae = bench["Naive baseline"]["quality_filtered"]["mae"]
    lgb_mae = bench["LightGBM (Optuna)"]["quality_filtered"]["mae"]

    # 6. artifacts ---------------------------------------------------------------
    model.save_model(str(MODELS / "lightgbm_model.txt"))
    (MODELS / "lightgbm_artifacts.json").write_text(json.dumps({
        "features": features, "categorical": CAT_COLS, "category_levels": levels,
        "target": "log1p(yield); prediction = (expm1(raw) + 1) * smearing[crop] - 1",
        "smearing": {k: round(float(v), 5) for k, v in smear.items()},
        "params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()},
        "num_boost_round": n_rounds,
    }, indent=1), encoding="utf-8")
    (MODELS / "quality_thresholds.json").write_text(json.dumps({
        "rule": f"yield > {IMPOSSIBLE_YIELD_FACTOR:g} x crop's maximum training yield (2001-2017)",
        "factor": IMPOSSIBLE_YIELD_FACTOR,
        "max_yield_threshold": {k: float(v) for k, v in thresholds.items()},
    }, indent=1), encoding="utf-8")
    evaluation = {
        "generated": date.today().isoformat(),
        "production_model": "LightGBM (Optuna)",
        "split": {"train_years": "2001-2017", "test_years": "2018-2020", "dev_split": "train 2001-2014 / validate 2015-2017",
                  "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_features": len(features)},
        "optuna": {"trials": len(study.trials), "best_dev_r2": round(study.best_value, 4)},
        "quality_rule": f"drop test rows with yield > {IMPOSSIBLE_YIELD_FACTOR:g}x the crop's training maximum",
        "benchmark": bench,
        "mae_reduction_vs_baseline_pct": round(100 * (base_mae - lgb_mae) / base_mae, 1),
        "error_analysis": {
            "flagged_rows": int(flagged.sum()),
            "flagged_pct_of_test": round(100 * flagged.mean(), 2),
            "flagged_share_of_squared_error_pct": round(100 * sq[flagged].sum() / sq.sum(), 1),
            "worst_errors": worst.to_dict("records"),
        },
        "baseline_description": "naive forecast: last-3-year mean yield of the same district + crop",
    }
    (MODELS / "evaluation.json").write_text(json.dumps(evaluation, indent=1, default=float), encoding="utf-8")

    print(json.dumps({k: {s: v[s]["r2"] for s in v} for k, v in bench.items()}, indent=1))
    print(f"MAE reduction vs baseline (filtered): {evaluation['mae_reduction_vs_baseline_pct']}%")
    print(f"done in {time.time() - t0:.0f}s; model file {(MODELS / 'lightgbm_model.txt').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
