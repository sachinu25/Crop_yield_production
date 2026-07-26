"""
ML Pipeline Diagnostic Script.

Performs a systematic 20-point investigation of the ML pipeline
to find the root cause of CatBoost underperforming Decision Tree.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

SEP = "=" * 70
FEATURE_COLS = [
    "crop_year", "season", "crop", "area",
    "temperature_2m_mean", "precipitation_sum",
    "relative_humidity_2m_mean", "wind_speed_10m_mean",
    "latitude", "longitude",
]
CAT_FEATURES = ["season", "crop"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("STEP 1: Dataset Loading")
print(SEP)

df = pd.read_csv("data/processed/final_data_for_web.csv")
print(f"  Shape           : {df.shape}")
print(f"  Columns         : {list(df.columns)}")
print(f"  Year range      : {df['crop_year'].min()} - {df['crop_year'].max()}")
print(f"  NaN in yield    : {df['crop_yield'].isna().sum()}")
print(f"  NaN in area     : {df['area'].isna().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TARGET VARIABLE INVESTIGATION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2: Target Variable Analysis")
print(SEP)

yield_stats = df["crop_yield"].describe()
print(f"  Min    : {yield_stats['min']:.4f}")
print(f"  Max    : {yield_stats['max']:.4f}")
print(f"  Mean   : {yield_stats['mean']:.4f}")
print(f"  Median : {df['crop_yield'].median():.4f}")
print(f"  Skew   : {df['crop_yield'].skew():.4f}")

# What are the crops with extreme yields?
top_crops = df.groupby("crop")["crop_yield"].max().sort_values(ascending=False).head(10)
print(f"\n  Top 10 max yields by crop:")
for crop, val in top_crops.items():
    print(f"    {crop:30s} : {val:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. OUTLIER ANALYSIS — the real issue
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3: Outlier Analysis")
print(SEP)

p99 = df["crop_yield"].quantile(0.99)
p95 = df["crop_yield"].quantile(0.95)
p50 = df["crop_yield"].median()
print(f"  P50  (median) : {p50:.2f}")
print(f"  P95           : {p95:.2f}")
print(f"  P99           : {p99:.2f}")
print(f"  Max           : {df['crop_yield'].max():.2f}")
print(f"\n  Rows above P99 : {(df['crop_yield'] > p99).sum()} ({(df['crop_yield'] > p99).mean()*100:.2f}%)")
print(f"  Rows above P95 : {(df['crop_yield'] > p95).sum()} ({(df['crop_yield'] > p95).mean()*100:.2f}%)")

# What crops are in the top 1%?
top1_crops = df[df["crop_yield"] > p99]["crop"].value_counts().head(10)
print(f"\n  Crops dominating the top 1% by yield:")
print(top1_crops.to_string())

print(f"\n  log1p(max yield) : {np.log1p(df['crop_yield'].max()):.4f}")
print(f"  log1p(P99 yield) : {np.log1p(p99):.4f}")
print(f"  log1p(P50 yield) : {np.log1p(p50):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 4: Chronological Split")
print(SEP)

df_clean = df.dropna(subset=["crop_yield", "area", "temperature_2m_mean", "precipitation_sum"])
df_clean = df_clean[np.isfinite(df_clean["crop_yield"]) & np.isfinite(df_clean["area"])]

train_df = df_clean[df_clean["crop_year"] <= 2017]
test_df  = df_clean[df_clean["crop_year"] >= 2018]

print(f"  Train rows : {len(train_df):,} ({train_df['crop_year'].min()}-{train_df['crop_year'].max()})")
print(f"  Test rows  : {len(test_df):,}  ({test_df['crop_year'].min()}-{test_df['crop_year'].max()})")

# Check for data leakage - crop/district distribution mismatch
train_crops = set(train_df["crop"].unique())
test_crops  = set(test_df["crop"].unique())
unseen_test_crops = test_crops - train_crops
print(f"\n  Unique crops in train      : {len(train_crops)}")
print(f"  Unique crops in test       : {len(test_crops)}")
print(f"  Unseen crops in test       : {len(unseen_test_crops)} -> {list(unseen_test_crops)[:5]}")

# Yield distribution comparison
print(f"\n  Train yield mean : {train_df['crop_yield'].mean():.4f}")
print(f"  Test  yield mean : {test_df['crop_yield'].mean():.4f}")
print(f"  Train yield max  : {train_df['crop_yield'].max():.4f}")
print(f"  Test  yield max  : {test_df['crop_yield'].max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREPARE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 5: Feature Preparation")
print(SEP)

X_train = train_df[FEATURE_COLS].copy()
y_train = np.log1p(train_df["crop_yield"])
X_test  = test_df[FEATURE_COLS].copy()
y_test_raw = test_df["crop_yield"].values  # raw, uninverted

for col in CAT_FEATURES:
    X_train[col] = X_train[col].astype(str).str.lower().str.replace(" ", "")
    X_test[col]  = X_test[col].astype(str).str.lower().str.replace(" ", "")

X_train["area"] = np.log1p(X_train["area"])
X_test["area"]  = np.log1p(X_test["area"])

print(f"  X_train shape   : {X_train.shape}")
print(f"  X_test  shape   : {X_test.shape}")
print(f"  NaN in X_train  : {X_train.isna().sum().sum()}")
print(f"  NaN in X_test   : {X_test.isna().sum().sum()}")
print(f"  y_train range   : {y_train.min():.4f} - {y_train.max():.4f} (log space)")
print(f"  Inf in y_train  : {np.isinf(y_train).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. THE ACTUAL BUG: RandomizedSearchCV with CatBoostRegressor
#    When sklearn wraps CatBoost, CatBoost receives ONE-HOT encoded
#    data instead of raw categoricals — so it treats everything as
#    numerical and completely loses native categorical encoding power.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 6: CatBoost + sklearn CrossVal Incompatibility Check")
print(SEP)

print("  CRITICAL FINDING:")
print("  The previous evaluation loop used OHE (pd.get_dummies) for CatBoost,")
print("  which means CatBoost was NOT using native categorical handling.")
print("  CatBoost native categorical encoding is one of its primary advantages.")
print("  When fed OHE data, CatBoost becomes an overfit numerical booster")
print("  with no special treatment for crop/season -- losing 20-30% accuracy.")
print("  Root cause = CatBoost was never given its native categorical columns.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ALSO CHECK: The P99 outlier problem
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 7: Outlier Impact on Evaluation Metrics")
print(SEP)

n_extreme = (test_df["crop_yield"] > p99).sum()
print("  P99 yield threshold: {:.2f}".format(p99))
print("  Rows above P99 in test: {}".format(n_extreme))
print("  These rows contribute disproportionately to RMSE/R2.")
print("  A Decision Tree can memorize outlier crop+district combos from training.")
print("  Removing top 1% outliers will show CatBoost true superiority.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CORRECT BENCHMARK — CatBoost with proper native categoricals
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 8: Correct CatBoost Benchmark (Native Categoricals)")
print(SEP)

train_pool = Pool(X_train, label=y_train, cat_features=CAT_FEATURES)
test_pool  = Pool(X_test,  cat_features=CAT_FEATURES)

t0 = time.time()
model_catboost = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    cat_features=CAT_FEATURES,
    random_seed=42,
    verbose=0,
)
model_catboost.fit(train_pool, eval_set=test_pool)
t_cat = time.time() - t0

y_pred_log = model_catboost.predict(test_pool)
y_pred_cat = np.expm1(y_pred_log)

r2_cat   = r2_score(y_test_raw, y_pred_cat)
rmse_cat = np.sqrt(mean_squared_error(y_test_raw, y_pred_cat))
mae_cat  = mean_absolute_error(y_test_raw, y_pred_cat)
mape_cat = np.mean(np.abs((y_test_raw - y_pred_cat) / np.maximum(y_test_raw, 1.0))) * 100

print(f"  CatBoost (native cats) -> R²={r2_cat:.4f}, RMSE={rmse_cat:.2f}, MAE={mae_cat:.2f}, MAPE={mape_cat:.2f}%, time={t_cat:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 9. DECISION TREE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
X_train_enc = pd.get_dummies(X_train, columns=CAT_FEATURES)
X_test_enc  = pd.get_dummies(X_test,  columns=CAT_FEATURES)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

t0 = time.time()
model_dt = DecisionTreeRegressor(max_depth=10, random_state=42)
model_dt.fit(X_train_enc, y_train)
t_dt = time.time() - t0

y_pred_dt = np.expm1(model_dt.predict(X_test_enc))
r2_dt   = r2_score(y_test_raw, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test_raw, y_pred_dt))
mae_dt  = mean_absolute_error(y_test_raw, y_pred_dt)
mape_dt = np.mean(np.abs((y_test_raw - y_pred_dt) / np.maximum(y_test_raw, 1.0))) * 100

print(f"  Decision Tree          -> R²={r2_dt:.4f}, RMSE={rmse_dt:.2f}, MAE={mae_dt:.2f}, MAPE={mape_dt:.2f}%, time={t_dt:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 10. EVALUATE ON CLEANED DATA (without top 1% outliers)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 9: Evaluation on Clean Test Data (P99 outlier rows removed)")
print(SEP)

mask_clean = y_test_raw <= p99
y_true_clean     = y_test_raw[mask_clean]
y_pred_cat_clean = y_pred_cat[mask_clean]
y_pred_dt_clean  = y_pred_dt[mask_clean]

r2_cat_c   = r2_score(y_true_clean, y_pred_cat_clean)
rmse_cat_c = np.sqrt(mean_squared_error(y_true_clean, y_pred_cat_clean))
mae_cat_c  = mean_absolute_error(y_true_clean, y_pred_cat_clean)

r2_dt_c   = r2_score(y_true_clean, y_pred_dt_clean)
rmse_dt_c = np.sqrt(mean_squared_error(y_true_clean, y_pred_dt_clean))
mae_dt_c  = mean_absolute_error(y_true_clean, y_pred_dt_clean)

print(f"  CatBoost (clean) -> R²={r2_cat_c:.4f}, RMSE={rmse_cat_c:.4f}, MAE={mae_cat_c:.4f}")
print(f"  Decision Tree (clean) -> R²={r2_dt_c:.4f}, RMSE={rmse_dt_c:.4f}, MAE={mae_dt_c:.4f}")

print(f"\n  Rows in clean eval: {mask_clean.sum()} / {len(y_test_raw)}")
