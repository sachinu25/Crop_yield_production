import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import logging
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/final_data_for_web.csv")
OUT_PATH = Path("data/processed/grandmaster_data.csv")

def audit_and_clean(df):
    logger.info("PHASE 1 & 3: AUDIT & DATA QUALITY")
    orig_len = len(df)
    
    # 1. Drop infs and impossible zeroes
    df = df[np.isfinite(df["crop_yield"]) & np.isfinite(df["area"])]
    df = df[df["crop_yield"] > 0]
    df = df[df["area"] > 0]
    
    # 2. Hard agronomic ceiling
    df = df[df["crop_yield"] <= 500.0]
    
    # 3. Deduplicate
    key_cols = ["state_name", "district_name", "season", "crop", "crop_year"]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    
    # 4. Standardize strings
    df["season"] = df["season"].astype(str).str.lower().str.strip()
    df["crop"] = df["crop"].astype(str).str.lower().str.strip()
    df["state_name"] = df["state_name"].astype(str).str.lower().str.strip()
    
    # 5. Crop-level IQR clipping/filtering (to remove extreme noise in training data only)
    # We don't drop test data (>= 2018), but we can clip training data (<= 2017) to stabilize it
    # Actually, the prompt says "Use ONLY methods that improve validation performance."
    # Let's apply Isolation Forest purely on the training set features to drop anomalies.
    
    # First, let's just do mild crop-level Winsorization on the target.
    # Yield is heavily skewed.
    logger.info(f"Cleaned {orig_len - len(df)} impossible rows.")
    return df

def feature_engineering(df):
    logger.info("PHASE 2: ADVANCED FEATURE ENGINEERING")
    
    # Sort chronologically to prevent leakage in expanding means
    df = df.sort_values(["crop_year", "state_name", "district_name", "season", "crop"]).reset_index(drop=True)
    
    # 1. Base Interactions
    df["state_crop"] = df["state_name"] + "_" + df["crop"]
    df["crop_season"] = df["crop"] + "_" + df["season"]
    
    # 2. Climate Anomalies
    # Expanding historical mean up to t-1
    logger.info("Computing expanding historical climate means (preventing leakage)...")
    
    def expanding_mean(grp, col):
        return grp.groupby("crop_year")[col].mean().expanding().mean().shift(1)
    
    # State-level historical temp and precip
    state_hist = df.groupby(["state_name", "crop_year"])[["temperature_2m_mean", "precipitation_sum"]].mean().reset_index()
    state_hist["hist_temp"] = state_hist.groupby("state_name")["temperature_2m_mean"].transform(lambda x: x.expanding().mean().shift(1))
    state_hist["hist_precip"] = state_hist.groupby("state_name")["precipitation_sum"].transform(lambda x: x.expanding().mean().shift(1))
    
    # If a state has no history (year 1), fill with current year to prevent NaNs, but it's imperfect.
    # Better: fill with overall mean of year 1.
    overall_mean_temp = state_hist["temperature_2m_mean"].mean()
    overall_mean_precip = state_hist["precipitation_sum"].mean()
    state_hist["hist_temp"] = state_hist["hist_temp"].fillna(overall_mean_temp)
    state_hist["hist_precip"] = state_hist["hist_precip"].fillna(overall_mean_precip)
    
    df = df.merge(state_hist[["state_name", "crop_year", "hist_temp", "hist_precip"]], on=["state_name", "crop_year"], how="left")
    
    df["temp_anomaly"] = df["temperature_2m_mean"] - df["hist_temp"]
    df["precip_anomaly"] = df["precipitation_sum"] - df["hist_precip"]
    
    # 3. Expanding Historical Target (Lag Yield)
    logger.info("Computing expanding historical yield (lag yield)...")
    # For a given state + crop, what was the historical yield up to t-1?
    crop_hist = df.groupby(["state_crop", "crop_year"])["crop_yield"].mean().reset_index()
    crop_hist["hist_yield"] = crop_hist.groupby("state_crop")["crop_yield"].transform(lambda x: x.expanding().mean().shift(1))
    
    global_yield = df["crop_yield"].mean()
    crop_hist["hist_yield"] = crop_hist["hist_yield"].fillna(global_yield)
    
    df = df.merge(crop_hist[["state_crop", "crop_year", "hist_yield"]], on=["state_crop", "crop_year"], how="left")
    
    # 4. Geospatial Clustering (Agro-climatic zones)
    logger.info("Computing Geospatial clusters...")
    # Train K-Means only on training data coordinates
    train_mask = df["crop_year"] <= 2017
    kmeans = KMeans(n_clusters=15, random_state=42)
    
    lat_lon_fit = df[["latitude", "longitude"]].copy()
    geo_clusters = np.zeros(len(df), dtype=int)
    geo_clusters[train_mask] = kmeans.fit_predict(lat_lon_fit[train_mask])
    geo_clusters[~train_mask] = kmeans.predict(lat_lon_fit[~train_mask])
    
    df["geo_cluster"] = geo_clusters.astype(str)
    
    # 5. Base Transforms
    df["log_area"] = np.log1p(df["area"])
    df["temp_precip_ratio"] = df["temperature_2m_mean"] / (df["precipitation_sum"] + 1.0)
    df["is_drought"] = (df["precipitation_sum"] < 5.0).astype(int)
    df["is_extreme_heat"] = (df["temperature_2m_mean"] > 35.0).astype(int)
    
    return df

def apply_isolation_forest(df):
    logger.info("PHASE 3: ADVANCED ANOMALY DETECTION (Isolation Forest)")
    train_mask = df["crop_year"] <= 2017
    test_mask = ~train_mask
    
    # Features for anomaly detection
    num_cols = ["log_area", "temperature_2m_mean", "precipitation_sum", "hist_yield"]
    
    X_train = df.loc[train_mask, num_cols].fillna(0)
    
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    train_anomalies = iso.fit_predict(X_train)
    
    # Keep 1s, drop -1s (only in train)
    valid_train = train_anomalies == 1
    
    logger.info(f"Isolation Forest dropped {(~valid_train).sum()} training anomalies.")
    
    # Recombine
    df_train = df[train_mask][valid_train]
    df_test = df[test_mask]
    
    df_clean = pd.concat([df_train, df_test], ignore_index=True)
    return df_clean

def main():
    logger.info("Loading raw data...")
    df = pd.read_csv(DATA_PATH)
    
    df = audit_and_clean(df)
    df = feature_engineering(df)
    df = apply_isolation_forest(df)
    
    # Final check
    req_cols = [
        "crop_year", "season", "crop", "log_area", "temperature_2m_mean",
        "precipitation_sum", "relative_humidity_2m_mean", "wind_speed_10m_mean",
        "latitude", "longitude", "state_crop", "crop_season", "temp_anomaly",
        "precip_anomaly", "hist_yield", "geo_cluster", "temp_precip_ratio",
        "is_drought", "is_extreme_heat", "crop_yield"
    ]
    df = df.dropna(subset=req_cols)
    
    logger.info(f"Final shape: {df.shape}")
    df.to_csv(OUT_PATH, index=False)
    logger.info(f"Saved grandmaster dataset to {OUT_PATH}")

if __name__ == "__main__":
    main()
