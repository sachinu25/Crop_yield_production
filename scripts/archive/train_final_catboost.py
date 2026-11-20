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





# ── Configuration ──
# optuna(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("ml_v4")

ROOT = Path(__file__).resolve().parent
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
    # (gbdt_train_path, index=False)
    # (cat_train_path, index=False)
    # (gbdt_test_path, index=False)
    # (cat_test_path, index=False)
    # _path, y_train)
    # _path, y_test_raw)
    if False:
        json.dump({"GBDT_FEATURES": GBDT_FEATURES, "CAT_FEATURES": CAT_FEATURES, "CATBOOST_CAT": CATBOOST_CAT}, f)


import json
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load best params
with open(MODELS / 'best_params.json', 'r') as f:
    best_params = json.load(f)['CatBoost']

print(f'Training CatBoost with params: {best_params}')

model = CatBoostRegressor(**best_params, cat_features=CATBOOST_CAT, random_seed=42, verbose=100)
model.fit(X_train_cat, y_train)

# Save model
model.save_model(MODELS / 'CatBoost_best_final.cbm')

# Predict
preds = np.expm1(model.predict(X_test_cat))

# Metrics
r2 = r2_score(y_test_raw, preds)
rmse = np.sqrt(mean_squared_error(y_test_raw, preds))
mae = mean_absolute_error(y_test_raw, preds)

print(f'CatBoost Hold-out R2: {r2}')
print(f'CatBoost Hold-out RMSE: {rmse}')
print(f'CatBoost Hold-out MAE: {mae}')

# Save predictions
preds_df = pd.DataFrame({'True_Yield': y_test_raw, 'Predicted_Yield': preds})
preds_df.to_csv(REPORTS / 'catboost_predictions.csv', index=False)

# Save report
report = f'''# CatBoost Final Report
- **Hold-out R2**: {r2}
- **Hold-out RMSE**: {rmse}
- **Hold-out MAE**: {mae}
'''
with open(REPORTS / 'catboost_report.md', 'w') as f:
    f.write(report)
