"""
ML V4 Resumable Pipeline (Fault-Tolerant)
====================================================
Objective: Maximize predictive performance (Target: R² > 0.80) with 100% fault tolerance.
"""

import time, warnings, json, logging, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from scipy.optimize import minimize

from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap

# ── Configuration ──
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("ml_v4")

ROOT = Path(__file__).resolve().parent.parent
DATA_IN = ROOT / "data" / "processed" / "final_data_for_web.csv"
REPORTS = ROOT / "reports"; REPORTS.mkdir(exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
CHECKPOINTS = ROOT / "checkpoints"; CHECKPOINTS.mkdir(exist_ok=True)

N_TRIALS = 200  # Will be overridden in test mode
N_FOLDS = 5
YIELD_CEILING = 500.0
RS = 42
SEP = "=" * 70

DB_URL = f"sqlite:///{CHECKPOINTS}/optuna_v4.db"

def hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

def calc_metrics(y_true, y_pred):
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
    }

# Check if test mode
if len(sys.argv) > 1 and sys.argv[1] == "--test":
    N_TRIALS = 2
    log.info("RUNNING IN TEST MODE: N_TRIALS=%d", N_TRIALS)

# ======================================================================
# [1/8] DATA LOADING & CACHING
# ======================================================================
log.info(SEP)
log.info("[1/8] DATA PROCESSING & CHECKPOINTING")
log.info(SEP)

gbdt_train_path = CHECKPOINTS / "X_train_gbdt.parquet"
cat_train_path = CHECKPOINTS / "X_train_cat.parquet"
gbdt_test_path = CHECKPOINTS / "X_test_gbdt.parquet"
cat_test_path = CHECKPOINTS / "X_test_cat.parquet"
y_train_path = CHECKPOINTS / "y_train.npy"
y_test_path = CHECKPOINTS / "y_test.npy"
meta_path = CHECKPOINTS / "metadata.json"

if all(p.exists() for p in [gbdt_train_path, cat_train_path, gbdt_test_path, cat_test_path, y_train_path, y_test_path, meta_path]):
    log.info("Found existing processed data checkpoints. Loading from disk...")
    X_train_gbdt = pd.read_parquet(gbdt_train_path)
    X_train_cat = pd.read_parquet(cat_train_path)
    X_test_gbdt = pd.read_parquet(gbdt_test_path)
    X_test_cat = pd.read_parquet(cat_test_path)
    y_train = np.load(y_train_path)
    y_test_raw = np.load(y_test_path)
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        GBDT_FEATURES = meta["GBDT_FEATURES"]
        CAT_FEATURES = meta["CAT_FEATURES"]
        CATBOOST_CAT = meta["CATBOOST_CAT"]
        
else:
    log.info("No checkpoints found. Running full preprocessing...")
    t0 = time.time()
    df = pd.read_csv(DATA_IN)
    df = df[np.isfinite(df["crop_yield"]) & np.isfinite(df["area"])]
    df = df[(df["crop_yield"] > 0) & (df["area"] > 0)]
    df = df[df["crop_yield"] <= YIELD_CEILING]

    for col in ["season", "crop", "state_name", "district_name"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    key_cols = ["state_name", "district_name", "season", "crop", "crop_year"]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    train_mask = df["crop_year"] <= 2017
    for crop_name in df.loc[train_mask, "crop"].unique():
        mask = train_mask & (df["crop"] == crop_name)
        vals = df.loc[mask, "crop_yield"]
        if len(vals) < 30: continue
        q1, q3 = vals.quantile(0.05), vals.quantile(0.95)
        iqr = q3 - q1
        lower, upper = max(0.001, q1 - 3.0 * iqr), q3 + 3.0 * iqr
        clipped = vals.clip(lower=lower, upper=upper)
        df.loc[mask, "crop_yield"] = clipped

    df = df.sort_values(["crop_year", "state_name", "district_name"]).reset_index(drop=True)
    df["state_crop"] = df["state_name"] + "_" + df["crop"]
    df["crop_season"] = df["crop"] + "_" + df["season"]
    df["district_crop"] = df["district_name"] + "_" + df["crop"]

    df["log_area"] = np.log1p(df["area"])
    df["log_precip"] = np.log1p(df["precipitation_sum"].clip(lower=0))

    df["temp_precip_ratio"] = df["temperature_2m_mean"] / (df["precipitation_sum"] + 1.0)
    df["precip_per_temp"] = df["precipitation_sum"] / (df["temperature_2m_mean"] + 1.0)
    df["temp_x_humidity"] = df["temperature_2m_mean"] * df["relative_humidity_2m_mean"] / 100.0
    df["precip_x_humidity"] = df["precipitation_sum"] * df["relative_humidity_2m_mean"] / 100.0
    df["wind_x_temp"] = df["wind_speed_10m_mean"] * df["temperature_2m_mean"] / 100.0
    df["humidity_per_wind"] = df["relative_humidity_2m_mean"] / (df["wind_speed_10m_mean"] + 1.0)
    df["gdd_proxy"] = np.maximum(0, df["temperature_2m_mean"] - 10.0)
    df["gdd_x_precip"] = df["gdd_proxy"] * df["log_precip"]
    df["temp_squared"] = df["temperature_2m_mean"] ** 2
    df["precip_squared"] = df["precipitation_sum"] ** 2

    df["is_drought"] = (df["precipitation_sum"] < 5.0).astype(int)
    df["is_extreme_heat"] = (df["temperature_2m_mean"] > 35.0).astype(int)
    df["is_high_humidity"] = (df["relative_humidity_2m_mean"] > 80.0).astype(int)

    df["years_since_2001"] = df["crop_year"] - 2001
    df["year_norm_sq"] = ((df["crop_year"] - 2001) / 19.0) ** 2
    season_map = {"kharif": 0, "rabi": 1, "whole year": 2, "summer": 3, "winter": 4, "autumn": 5}
    season_ord = df["season"].map(season_map).fillna(2)
    df["season_sin"] = np.sin(2 * np.pi * season_ord / 6)
    df["season_cos"] = np.cos(2 * np.pi * season_ord / 6)

    global_yield = df.loc[df["crop_year"] <= 2017, "crop_yield"].mean()
    def compute_lags(frame, group_col):
        grp = frame.groupby([group_col, "crop_year"])["crop_yield"].mean().reset_index()
        grp[f"exp_mean_{group_col}"] = grp.groupby(group_col)["crop_yield"].transform(lambda x: x.expanding().mean().shift(1)).fillna(global_yield)
        grp[f"roll3_mean_{group_col}"] = grp.groupby(group_col)["crop_yield"].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)).fillna(global_yield)
        grp[f"ewma_{group_col}"] = grp.groupby(group_col)["crop_yield"].transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean().shift(1)).fillna(global_yield)
        merge_cols = [group_col, "crop_year", f"exp_mean_{group_col}", f"roll3_mean_{group_col}", f"ewma_{group_col}"]
        return frame.merge(grp[merge_cols], on=[group_col, "crop_year"], how="left")

    for col in ["state_crop", "district_crop", "crop", "state_name"]:
        df = compute_lags(df, col)
    df["yield_trend_sc"] = df["exp_mean_state_crop"] - df["roll3_mean_state_crop"]

    scl = df.groupby(["state_name", "crop_year"])[["temperature_2m_mean", "precipitation_sum"]].mean().reset_index()
    scl["ht"] = scl.groupby("state_name")["temperature_2m_mean"].transform(lambda x: x.expanding().mean().shift(1)).fillna(scl["temperature_2m_mean"].mean())
    scl["hp"] = scl.groupby("state_name")["precipitation_sum"].transform(lambda x: x.expanding().mean().shift(1)).fillna(scl["precipitation_sum"].mean())
    df = df.merge(scl[["state_name", "crop_year", "ht", "hp"]], on=["state_name", "crop_year"], how="left")
    df["temp_anomaly"] = df["temperature_2m_mean"] - df["ht"]
    df["precip_anomaly"] = df["precipitation_sum"] - df["hp"]
    df.drop(columns=["ht", "hp"], inplace=True)

    train_mask = df["crop_year"] <= 2017.0
    ll = df[["latitude", "longitude"]].values
    for k in [10, 30, 50]:
        km = KMeans(n_clusters=k, random_state=RS, n_init=5)
        geo = np.zeros(len(df), dtype=int)
        geo[train_mask] = km.fit_predict(ll[train_mask])
        geo[~train_mask] = km.predict(ll[~train_mask])
        df[f"geo_cluster_{k}"] = geo

    climate_cols = ["temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean", "wind_speed_10m_mean"]
    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=RS)
    c_train = scaler.fit_transform(df.loc[train_mask, climate_cols])
    pca.fit(c_train)
    pca_feats = pca.transform(scaler.transform(df[climate_cols]))
    df["climate_pca_1"] = pca_feats[:, 0]
    df["climate_pca_2"] = pca_feats[:, 1]

    for col in ["district_name", "state_name", "crop", "season"]:
        freq = df.loc[train_mask, col].value_counts()
        df[f"freq_{col}"] = df[col].map(freq).fillna(0)

    train_df = df[df["crop_year"] <= 2017].copy().reset_index(drop=True)
    test_df = df[df["crop_year"] >= 2018].copy().reset_index(drop=True)
    
    y_train_raw = train_df["crop_yield"].values.astype(np.float64)
    y_test_raw = test_df["crop_yield"].values.astype(np.float64)
    y_train = np.log1p(y_train_raw)

    te_cols = ["crop", "state_name", "district_name", "state_crop", "district_crop"]
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    for col in te_cols:
        looe_col = f"looe_{col}"
        train_df[looe_col] = np.nan
        global_mean = y_train.mean()
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            stats = pd.Series(y_train[tr_idx]).groupby(X_tr[col].values).mean()
            train_df.loc[val_idx, looe_col] = X_val[col].map(stats)
        train_df[looe_col] = train_df[looe_col].fillna(global_mean)
        stats_full = pd.Series(y_train).groupby(train_df[col].values).mean()
        test_df[looe_col] = test_df[col].map(stats_full).fillna(global_mean)

    LABEL_CATS = ["season", "crop", "state_name", "geo_cluster_10", "geo_cluster_30", "geo_cluster_50"]
    for col in LABEL_CATS:
        le = LabelEncoder()
        le_col = f"{col}_le"
        train_df[le_col] = le.fit_transform(train_df[col].astype(str))
        known = set(le.classes_)
        test_df[le_col] = le.transform(test_df[col].astype(str).map(lambda x: x if x in known else le.classes_[0]))

    CATBOOST_CAT = ["season", "crop", "state_name", "crop_season", "state_crop", "geo_cluster_10", "geo_cluster_30", "geo_cluster_50"]
    for cdf in [train_df, test_df]:
        for c in CATBOOST_CAT:
            cdf[c] = cdf[c].astype(str)

    NUM_FEATURES = [
        "crop_year", "log_area", "log_precip",
        "temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean", "wind_speed_10m_mean",
        "latitude", "longitude",
        "temp_precip_ratio", "precip_per_temp", "temp_x_humidity", "precip_x_humidity", "wind_x_temp", "humidity_per_wind",
        "is_drought", "is_extreme_heat", "is_high_humidity",
        "gdd_proxy", "gdd_x_precip", "temp_squared", "precip_squared",
        "years_since_2001", "year_norm_sq", "season_sin", "season_cos",
        "exp_mean_state_crop", "roll3_mean_state_crop", "ewma_state_crop",
        "exp_mean_district_crop", "roll3_mean_district_crop", "ewma_district_crop",
        "exp_mean_crop", "roll3_mean_crop", "ewma_crop",
        "exp_mean_state_name", "roll3_mean_state_name", "ewma_state_name",
        "yield_trend_sc", "temp_anomaly", "precip_anomaly",
        "climate_pca_1", "climate_pca_2",
        "freq_district_name", "freq_state_name", "freq_crop", "freq_season",
        "looe_crop", "looe_state_name", "looe_district_name", "looe_state_crop", "looe_district_crop"
    ]
    CAT_LE = [f"{c}_le" for c in LABEL_CATS]
    GBDT_FEATURES = NUM_FEATURES + CAT_LE
    CAT_FEATURES = NUM_FEATURES + CATBOOST_CAT

    X_train_gbdt = train_df[GBDT_FEATURES].fillna(0)
    X_test_gbdt = test_df[GBDT_FEATURES].fillna(0)
    X_train_cat = train_df[CAT_FEATURES].fillna(0)
    X_test_cat = test_df[CAT_FEATURES].fillna(0)

    # Save Checkpoints
    log.info("Saving preprocessed checkpoints...")
    X_train_gbdt.to_parquet(gbdt_train_path, index=False)
    X_train_cat.to_parquet(cat_train_path, index=False)
    X_test_gbdt.to_parquet(gbdt_test_path, index=False)
    X_test_cat.to_parquet(cat_test_path, index=False)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test_raw)
    with open(meta_path, "w") as f:
        json.dump({"GBDT_FEATURES": GBDT_FEATURES, "CAT_FEATURES": CAT_FEATURES, "CATBOOST_CAT": CATBOOST_CAT}, f)

tscv = TimeSeriesSplit(n_splits=N_FOLDS)
folds_gbdt = list(tscv.split(X_train_gbdt))
folds_cat = list(tscv.split(X_train_cat))


def optimize_model(study_name, objective, n_trials):
    log.info(SEP)
    log.info(f"OPTUNA: {study_name} ({n_trials} trials, DB Checkpoint)")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=DB_URL,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20)
    )
    
    completed = len(study.trials)
    if completed >= n_trials:
        log.info(f"Found {completed}/{n_trials} completed trials for {study_name}. Skipping optimization...")
    else:
        remaining = n_trials - completed
        log.info(f"Found {completed} trials for {study_name}. Resuming from Trial {completed + 1} ({remaining} remaining)...")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ======================================================================
# [2/8] OPTUNA: LightGBM
# ======================================================================
def lgb_objective(trial):
    p = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 100) if N_TRIALS < 5 else trial.suggest_int("n_estimators", 1000, 6000),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 1024),
        "max_depth":         trial.suggest_int("max_depth", 5, 20),
        "learning_rate":     trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 20.0, log=True),
        "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 20.0, log=True),
        "max_bin":           trial.suggest_int("max_bin", 127, 1023),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    }
    scores = []
    for fi, (tr_i, va_i) in enumerate(folds_gbdt):
        m = lgb.LGBMRegressor(**p, random_state=RS, verbosity=-1, n_jobs=-1)
        m.fit(X_train_gbdt.iloc[tr_i], y_train[tr_i], eval_set=[(X_train_gbdt.iloc[va_i], y_train[va_i])], callbacks=[lgb.early_stopping(100, verbose=False)])
        scores.append(np.sqrt(mean_squared_error(y_train[va_i], m.predict(X_train_gbdt.iloc[va_i]))))
        trial.report(np.mean(scores), fi)
        if trial.should_prune(): raise optuna.TrialPruned()
    return np.mean(scores)

lgb_params, lgb_score = optimize_model("LightGBM", lgb_objective, N_TRIALS)

# ======================================================================
# [3/8] OPTUNA: CatBoost
# ======================================================================
def cat_objective(trial):
    p = {
        "iterations":          trial.suggest_int("iterations", 50, 100) if N_TRIALS < 5 else trial.suggest_int("iterations", 1000, 3000), 
        "depth":               trial.suggest_int("depth", 4, 8),            
        "learning_rate":       trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 0.1, 50.0, log=True),
        "subsample":           trial.suggest_float("subsample", 0.5, 1.0),
        "random_strength":     trial.suggest_float("random_strength", 0.1, 20.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count":        trial.suggest_int("border_count", 32, 255),
    }
    scores = []
    for fi, (tr_i, va_i) in enumerate(folds_cat):
        tr_pool = Pool(X_train_cat.iloc[tr_i], y_train[tr_i], cat_features=CATBOOST_CAT)
        va_pool = Pool(X_train_cat.iloc[va_i], y_train[va_i], cat_features=CATBOOST_CAT)
        m = CatBoostRegressor(**p, random_seed=RS, verbose=0, early_stopping_rounds=100, thread_count=-1)
        m.fit(tr_pool, eval_set=va_pool)
        scores.append(np.sqrt(mean_squared_error(y_train[va_i], m.predict(va_pool))))
        trial.report(np.mean(scores), fi)
        if trial.should_prune(): raise optuna.TrialPruned()
    return np.mean(scores)

cat_params, cat_score = optimize_model("CatBoost", cat_objective, N_TRIALS)

# ======================================================================
# [4/8] OPTUNA: XGBoost
# ======================================================================
def xgb_objective(trial):
    p = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 100) if N_TRIALS < 5 else trial.suggest_int("n_estimators", 1000, 6000),
        "max_depth":        trial.suggest_int("max_depth", 4, 15),
        "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 150),
        "gamma":            trial.suggest_float("gamma", 0.0, 10.0),
        "max_bin":          trial.suggest_int("max_bin", 128, 512),
    }
    scores = []
    for fi, (tr_i, va_i) in enumerate(folds_gbdt):
        m = xgb.XGBRegressor(**p, random_state=RS, verbosity=0, n_jobs=-1, early_stopping_rounds=100, tree_method="hist")
        m.fit(X_train_gbdt.iloc[tr_i], y_train[tr_i], eval_set=[(X_train_gbdt.iloc[va_i], y_train[va_i])], verbose=False)
        scores.append(np.sqrt(mean_squared_error(y_train[va_i], m.predict(X_train_gbdt.iloc[va_i]))))
        trial.report(np.mean(scores), fi)
        if trial.should_prune(): raise optuna.TrialPruned()
    return np.mean(scores)

xgb_params, xgb_score = optimize_model("XGBoost", xgb_objective, N_TRIALS)

log.info("Optimization Complete. Finalizing models...")
