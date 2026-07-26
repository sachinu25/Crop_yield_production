"""
Prediction service — loads CatBoost model and runs crop yield inference.

This module replaces the `predicction()` function (note: typo fixed) that was
duplicated in both Flask helper.py and Streamlit Helper/helper.py.

Bugs fixed:
    - Typo: predicction -> prediction (was in both files)
    - Consistent area handling: both apps now use the same transform
    - Model loaded once and reused (not re-loaded on every import)

Design decisions:
    - Model is loaded lazily on first call, not at module import time.
      This prevents import-time crashes if the model file is missing
      and allows better error messages.
    - Added comprehensive feature engineering for the 60-feature model.
"""

import logging
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.config.settings import settings, PROJECT_ROOT
from src.utils.exceptions import PredictionError

logger = logging.getLogger(__name__)

# Module-level caches (lazy-loaded)
_model: Optional[CatBoostRegressor] = None
_lookup: Optional[pd.DataFrame] = None
_pca_info: Optional[Dict] = None


def _load_resources():
    """Load model, lookup table, and PCA info."""
    global _model, _lookup, _pca_info
    
    if _model is None:
        model_path = str(settings.MODEL_PATH)
        logger.info("Loading CatBoost model from %s", model_path)
        try:
            _model = CatBoostRegressor()
            _model.load_model(model_path)
        except Exception as exc:
            raise PredictionError(f"Could not load CatBoost model from {model_path}") from exc
            
    if _lookup is None:
        try:
            lookup_path = PROJECT_ROOT / "models" / "inference_lookup.parquet"
            _lookup = pd.read_parquet(lookup_path)
        except Exception as exc:
            raise PredictionError("Could not load inference_lookup.parquet") from exc
            
    if _pca_info is None:
        try:
            pca_path = PROJECT_ROOT / "models" / "pca_info.json"
            with open(pca_path, 'r') as f:
                _pca_info = json.load(f)
        except Exception as exc:
            raise PredictionError("Could not load pca_info.json") from exc

    return _model, _lookup, _pca_info


def predict_yield(data: Dict[str, Any]) -> Dict[str, float]:
    """Predict crop yield for one or more crops (Flask app)."""
    model, _, _ = _load_resources()
    predictions: Dict[str, float] = {}

    for crop_name in data["crop"]:
        try:
            input_df = _build_input_df(data, crop_name)
            
            # The model expects exactly these categorical features
            CATBOOST_CAT = ['season', 'crop', 'state_name', 'crop_season', 'state_crop', 'geo_cluster_10', 'geo_cluster_30', 'geo_cluster_50']
            for c in CATBOOST_CAT:
                input_df[c] = input_df[c].astype(str)
                
            input_pool = Pool(input_df, cat_features=CATBOOST_CAT)
            pred_log = model.predict(input_pool)
            pred = float(np.expm1(pred_log[0]))
            predictions[crop_name] = round(pred, 3)
        except Exception as exc:
            logger.error("Prediction failed for crop '%s': %s", crop_name, exc)
            raise PredictionError(f"Prediction failed for crop '{crop_name}'") from exc

    return predictions


def predict_yield_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Predict crop yield for one or more crops (Streamlit app)."""
    model, _, _ = _load_resources()
    results: List[Dict[str, Any]] = []

    for crop_name in data["crop"]:
        try:
            input_df = _build_input_df(data, crop_name)
            
            CATBOOST_CAT = ['season', 'crop', 'state_name', 'crop_season', 'state_crop', 'geo_cluster_10', 'geo_cluster_30', 'geo_cluster_50']
            for c in CATBOOST_CAT:
                input_df[c] = input_df[c].astype(str)
                
            input_pool = Pool(input_df, cat_features=CATBOOST_CAT)
            pred_log = model.predict(input_pool)
            pred = float(np.expm1(pred_log[0]))
            results.append({"Crop": crop_name, "Yield": round(pred, 3)})
        except Exception as exc:
            logger.error("Prediction failed for crop '%s': %s", crop_name, exc)
            raise PredictionError(f"Prediction failed for crop '{crop_name}'") from exc

    return results


def _build_input_df(data: Dict[str, Any], crop_name: str) -> pd.DataFrame:
    """Build the full 60-feature DataFrame for CatBoost inference."""
    _, lookup, pca_info = _load_resources()
    
    # 1. Base values
    cy = data["crop_year"]
    sn = str(data.get("state_name", data.get("state"))).lower().strip()
    dn = str(data.get("district_name", data.get("district"))).lower().strip()
    ssn = str(data["season"]).lower().strip()
    cn = crop_name.lower().strip()
    
    row = {
        'crop_year': cy,
        'state_name': sn,
        'district_name': dn,
        'season': ssn,
        'crop': cn,
        'area': data["area"],
        'temperature_2m_mean': data["temperature_2m_mean"],
        'precipitation_sum': data["precipitation_sum"],
        'relative_humidity_2m_mean': data["relative_humidity_2m_mean"],
        'wind_speed_10m_mean': data["wind_speed_10m_mean"]
    }
    df = pd.DataFrame([row])
    
    # 2. Join static features from lookup table
    # We join on state_name, district_name, crop, season
    match = lookup[
        (lookup['state_name'] == sn) & 
        (lookup['district_name'] == dn) & 
        (lookup['crop'] == cn) & 
        (lookup['season'] == ssn)
    ]
    
    if match.empty:
        # Fallback if specific combo doesn't exist
        match = lookup[
            (lookup['state_name'] == sn) & 
            (lookup['district_name'] == dn) & 
            (lookup['crop'] == cn)
        ]
    if match.empty:
        match = lookup[
            (lookup['state_name'] == sn) & 
            (lookup['crop'] == cn)
        ]
    if match.empty:
        # Extreme fallback: just get any row for this state to fill in static stuff
        match = lookup[lookup['state_name'] == sn]
        
    if match.empty:
        raise PredictionError(f"No historical data found for state {sn}")
        
    match = match.iloc[0:1].reset_index(drop=True)
    
    # Copy all lookup columns into our row (excluding the join keys to avoid overlap issues)
    for col in match.columns:
        if col not in df.columns:
            df[col] = match[col].values[0]
            
    # If the match was a fallback, fix up the dynamic static cols
    df['crop_season'] = df['crop'] + '_' + df['season']
    df['state_crop'] = df['state_name'] + '_' + df['crop']
    
    # 3. Dynamic math features
    df['log_area'] = np.log1p(df['area'])
    df['log_precip'] = np.log1p(df['precipitation_sum'].clip(lower=0))
    df['temp_precip_ratio'] = df['temperature_2m_mean'] / (df['precipitation_sum'] + 1.0)
    df['precip_per_temp'] = df['precipitation_sum'] / (df['temperature_2m_mean'] + 1.0)
    df['temp_x_humidity'] = df['temperature_2m_mean'] * df['relative_humidity_2m_mean'] / 100.0
    df['precip_x_humidity'] = df['precipitation_sum'] * df['relative_humidity_2m_mean'] / 100.0
    df['wind_x_temp'] = df['wind_speed_10m_mean'] * df['temperature_2m_mean'] / 100.0
    df['humidity_per_wind'] = df['relative_humidity_2m_mean'] / (df['wind_speed_10m_mean'] + 1.0)
    df['gdd_proxy'] = np.maximum(0, df['temperature_2m_mean'] - 10.0)
    df['gdd_x_precip'] = df['gdd_proxy'] * df['log_precip']
    df['temp_squared'] = df['temperature_2m_mean'] ** 2
    df['precip_squared'] = df['precipitation_sum'] ** 2
    df['is_drought'] = (df['precipitation_sum'] < 5.0).astype(int)
    df['is_extreme_heat'] = (df['temperature_2m_mean'] > 35.0).astype(int)
    df['is_high_humidity'] = (df['relative_humidity_2m_mean'] > 80.0).astype(int)
    df['years_since_2001'] = df['crop_year'] - 2001
    df['year_norm_sq'] = ((df['crop_year'] - 2001) / 19.0) ** 2
    
    # 4. PCA
    climate_cols = ['temperature_2m_mean', 'precipitation_sum', 'relative_humidity_2m_mean', 'wind_speed_10m_mean']
    raw_climate = df[climate_cols].values[0]
    mean = np.array(pca_info['scaler_mean'])
    scale = np.array(pca_info['scaler_scale'])
    components = np.array(pca_info['pca_components'])
    
    z = (raw_climate - mean) / scale
    pca_feats = z @ components.T
    df['climate_pca_1'] = pca_feats[0]
    df['climate_pca_2'] = pca_feats[1]
    
    # 5. Order columns
    CATBOOST_CAT = ['season', 'crop', 'state_name', 'crop_season', 'state_crop', 'geo_cluster_10', 'geo_cluster_30', 'geo_cluster_50']
    NUM_FEATURES = [
        'crop_year', 'log_area', 'log_precip',
        'temperature_2m_mean', 'precipitation_sum', 'relative_humidity_2m_mean', 'wind_speed_10m_mean',
        'latitude', 'longitude',
        'temp_precip_ratio', 'precip_per_temp', 'temp_x_humidity', 'precip_x_humidity', 'wind_x_temp', 'humidity_per_wind',
        'is_drought', 'is_extreme_heat', 'is_high_humidity',
        'gdd_proxy', 'gdd_x_precip', 'temp_squared', 'precip_squared',
        'years_since_2001', 'year_norm_sq', 'season_sin', 'season_cos',
        'exp_mean_state_crop', 'roll3_mean_state_crop', 'ewma_state_crop',
        'exp_mean_district_crop', 'roll3_mean_district_crop', 'ewma_district_crop',
        'exp_mean_crop', 'roll3_mean_crop', 'ewma_crop',
        'exp_mean_state_name', 'roll3_mean_state_name', 'ewma_state_name',
        'yield_trend_sc', 'temp_anomaly', 'precip_anomaly',
        'climate_pca_1', 'climate_pca_2',
        'freq_district_name', 'freq_state_name', 'freq_crop', 'freq_season',
        'looe_crop', 'looe_state_name', 'looe_district_name', 'looe_state_crop', 'looe_district_crop'
    ]
    CAT_FEATURES = NUM_FEATURES + CATBOOST_CAT
    
    final_df = df[CAT_FEATURES].copy()
    final_df = final_df.fillna(0)
    return final_df
