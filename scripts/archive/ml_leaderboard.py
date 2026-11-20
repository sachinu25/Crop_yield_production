import time
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import json

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/final_data_for_web.csv")
YIELD_CEILING_THA = 500.0

FEATURE_COLS = [
    "crop_year", "season", "crop", "area",
    "temperature_2m_mean", "precipitation_sum",
    "relative_humidity_2m_mean", "wind_speed_10m_mean",
    "latitude", "longitude",
]
CAT_FEATURES = ["season", "crop"]

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[np.isfinite(df["crop_yield"]) & np.isfinite(df["area"])]
    df = df[df["crop_yield"] > 0]
    df = df[df["crop_yield"] <= YIELD_CEILING_THA]
    df = df.drop_duplicates(subset=["state_name", "district_name", "season", "crop", "crop_year"], keep="first")
    df = df.dropna(subset=FEATURE_COLS + ["crop_yield"])
    
    df["season"] = df["season"].astype(str).str.lower().str.strip()
    df["crop"]   = df["crop"].astype(str).str.lower().str.strip()
    
    df["temp_precip_ratio"] = df["temperature_2m_mean"] / (df["precipitation_sum"] + 1.0)
    df["is_drought"] = (df["precipitation_sum"] < 5.0).astype(int)
    df["is_extreme_heat"] = (df["temperature_2m_mean"] > 35.0).astype(int)
    df["log_area"] = np.log1p(df["area"])
    
    df = df.sort_values("crop_year").reset_index(drop=True)
    return df

def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100
    return rmse, mae, r2, mape

def main():
    df = load_data()
    
    FEATURE_COLS_ENG = [
        "crop_year", "season", "crop", "log_area",
        "temperature_2m_mean", "precipitation_sum",
        "relative_humidity_2m_mean", "wind_speed_10m_mean",
        "latitude", "longitude",
        "temp_precip_ratio", "is_drought", "is_extreme_heat",
    ]
    
    X = df[FEATURE_COLS_ENG].copy()
    y_raw = df["crop_yield"].values
    y = np.log1p(y_raw)
    
    # Pre-encode for sklearn models
    X_ohe = pd.get_dummies(X, columns=CAT_FEATURES)
    
    models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "needs_ohe": True,
            "params": {}
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(max_depth=12, random_state=42),
            "needs_ohe": True,
            "params": {"max_depth": 12}
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42),
            "needs_ohe": True,
            "params": {"n_estimators": 50, "max_depth": 12}
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42),
            "needs_ohe": True,
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1),
            "needs_ohe": True,
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        },
        "CatBoost": {
            "model": CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1, cat_features=CAT_FEATURES, verbose=0, random_seed=42),
            "needs_ohe": False,
            "params": {"iterations": 300, "depth": 6, "learning_rate": 0.1}
        }
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    results = []
    
    for name, m_info in models.items():
        model = m_info["model"]
        needs_ohe = m_info["needs_ohe"]
        X_curr = X_ohe if needs_ohe else X
        
        cv_r2_scores = []
        
        for train_idx, test_idx in tscv.split(X_curr):
            X_train, X_test = X_curr.iloc[train_idx], X_curr.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_test_raw_cv = y_raw[test_idx]
            
            if name == "CatBoost":
                train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
                model.fit(train_pool)
                y_pred = np.expm1(model.predict(Pool(X_test, cat_features=CAT_FEATURES)))
            else:
                model.fit(X_train, y_train)
                y_pred = np.expm1(model.predict(X_test))
                
            _, _, r2_cv, _ = get_metrics(y_test_raw_cv, y_pred)
            cv_r2_scores.append(r2_cv)
            
        cv_mean = np.mean(cv_r2_scores)
        cv_std = np.std(cv_r2_scores)
        
        # Train on full train (up to 2017) and evaluate on test (2018+)
        train_mask = df["crop_year"] <= 2017
        test_mask = df["crop_year"] >= 2018
        
        X_tr, X_te = X_curr[train_mask], X_curr[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        y_te_raw = y_raw[test_mask]
        
        t0 = time.time()
        if name == "CatBoost":
            train_pool = Pool(X_tr, y_tr, cat_features=CAT_FEATURES)
            model.fit(train_pool)
        else:
            model.fit(X_tr, y_tr)
        t_train = time.time() - t0
        
        t1 = time.time()
        if name == "CatBoost":
            y_pred = np.expm1(model.predict(Pool(X_te, cat_features=CAT_FEATURES)))
        else:
            y_pred = np.expm1(model.predict(X_te))
        t_pred = time.time() - t1
        
        rmse, mae, r2, mape = get_metrics(y_te_raw, y_pred)
        
        results.append({
            "Model": name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE": round(mape, 2),
            "R2": round(r2, 4),
            "CV_mean": round(cv_mean, 4),
            "CV_std": round(cv_std, 4),
            "TrainTime": round(t_train, 2),
            "PredTime": round(t_pred, 4),
            "N_Features": X_tr.shape[1],
            "Params": json.dumps(m_info["params"])
        })
        
        print(f"Finished {name}: R2 = {r2:.4f}, CV_mean = {cv_mean:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv("reports/final_leaderboard.csv", index=False)
    print("Leaderboard saved to reports/final_leaderboard.csv")

if __name__ == "__main__":
    main()
