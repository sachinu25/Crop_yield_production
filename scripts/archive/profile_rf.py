import time, psutil, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Basic logging
import sys
sys.stdout.reconfigure(line_buffering=True)

print("Loading data...")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "grandmaster_data.csv"

df = pd.read_csv(DATA)
CATBOOST_CAT = ["season", "crop", "crop_season", "state_crop", "geo_cluster"]
SKLEARN_CAT = ["season", "crop", "geo_cluster"]
NUM_COLS = [
    "crop_year", "log_area", "temperature_2m_mean", "precipitation_sum",
    "relative_humidity_2m_mean", "wind_speed_10m_mean", "latitude", "longitude",
    "temp_precip_ratio", "is_drought", "is_extreme_heat",
    "temp_anomaly", "precip_anomaly", "hist_yield",
]

train_df = df[df["crop_year"] <= 2017].copy()
test_df  = df[df["crop_year"] >= 2018].copy()

y_train = np.log1p(train_df["crop_yield"].values).astype(np.float32)

print("One-hot encoding...")
t0 = time.time()
X_train_ohe = pd.get_dummies(train_df[NUM_COLS + SKLEARN_CAT].copy(), columns=SKLEARN_CAT)
# Convert to float32 contiguous array for sklearn efficiency
X_train_arr = np.ascontiguousarray(X_train_ohe.values, dtype=np.float32)
print(f"OHE done in {time.time()-t0:.2f}s. Shape: {X_train_arr.shape}")

tscv = TimeSeriesSplit(n_splits=3)
folds = list(tscv.split(X_train_arr))

# Pre-slice folds to avoid slicing overhead during the loop
fold_data = []
for tr_idx, va_idx in folds:
    fold_data.append({
        'X_tr': X_train_arr[tr_idx],
        'y_tr': y_train[tr_idx],
        'X_va': X_train_arr[va_idx],
        'y_va': y_train[va_idx]
    })

print("Profiling Trial 1: RandomForest(n_estimators=300, max_depth=15)")
model = RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=42)

process = psutil.Process(os.getpid())
process.cpu_percent(interval=None) # Start CPU profiling

total_fit_time = 0
total_pred_time = 0

for i, data in enumerate(fold_data):
    print(f"\nFold {i+1}...")
    
    t_fit_start = time.time()
    model.fit(data['X_tr'], data['y_tr'])
    t_fit = time.time() - t_fit_start
    total_fit_time += t_fit
    
    t_pred_start = time.time()
    preds = model.predict(data['X_va'])
    t_pred = time.time() - t_pred_start
    total_pred_time += t_pred
    
    cpu_util = process.cpu_percent(interval=None)
    mem_mb = process.memory_info().rss / (1024 * 1024)
    rmse = np.sqrt(mean_squared_error(data['y_va'], preds))
    
    print(f"  Fit time: {t_fit:.2f}s")
    print(f"  Pred time: {t_pred:.2f}s")
    print(f"  CPU Util: {cpu_util:.1f}%")
    print(f"  Memory: {mem_mb:.1f} MB")
    print(f"  RMSE: {rmse:.4f}")

total_time = total_fit_time + total_pred_time
print(f"\nTotal Trial Time: {total_time:.2f}s (Fit: {total_fit_time:.2f}s, Pred: {total_pred_time:.2f}s)")

if total_time > 300: # 5 minutes
    print("WARNING: Trial exceeded 5 minutes!")
    
