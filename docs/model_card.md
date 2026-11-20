# 📄 Model Card — CatBoost Crop Yield Regressor

## 1. Model Details
- **Developer**: Indian Agriculture ML Project
- **Model Type**: Gradient Boosted Decision Trees (CatBoost Regressor)
- **Model Version**: v1.0.0
- **Model Artifact**: `models/catboost_best_model.cbm`
- **Target Variable**: `np.log1p(crop_yield)` (Inverse log-transformed during inference)

## 2. Intended Use
- **Primary Use**: Estimating expected crop yield (tonnes/hectare) in Indian agricultural districts based on crop choice, season, area, and seasonal weather parameters.
- **Target Audience**: Agricultural analysts, data scientists, policy researchers, and agricultural planners.
- **Out of Scope**: Extreme catastrophic disaster forecasting without real-time weather API input.

## 3. Training & Evaluation Setup
- **Dataset Size**: 247,297 records after cleaning (2001–2020)
- **Train/Test Split Strategy**: **Chronological Time-Series Split**
  - **Training Set**: 2001–2017 (211,683 records)
  - **Testing Set**: 2018–2020 (35,614 records)
- **Categorical Features**: `season`, `crop` (native CatBoost categorical encoding)
- **Hyperparameters**:
  - `iterations`: 500
  - `learning_rate`: 0.08
  - `depth`: 6
  - `loss_function`: RMSE

## 4. Model Performance Metrics (Out-of-Time Test Set: 2018–2020)

Verified by re-running the evaluation (see `experiments/RESULTS.md`):

| Test set | Model | R² | MAE | RMSE |
|---|---|---|---|---|
| Full (35,614 rows) | CatBoost (notebook 07) | 0.605 | 1.42 | 10.01 |
| Full | Naive baseline (last-3-year district+crop mean) | 0.580 | 1.54 | 10.33 |
| Quality-filtered (35,479 rows) | LightGBM, log target + per-crop bias correction | 0.900 | 1.05 | 4.17 |
| Quality-filtered | Naive baseline | 0.864 | 1.18 | 4.86 |

The quality filter drops 135 test rows (0.38%) whose yield exceeds 2× that crop's training maximum. The threshold uses training data only. These rows cause 83% of the squared error on the full test set.

## 5. Feature Importance (SHAP Analysis Summary)

```
1. Crop Type (crop)                   ████████████████████ 42.5%
2. Cultivated Area (log_area)        ████████████ 25.1%
3. Season (season)                   ███████ 14.8%
4. Precipitation (precipitation_sum) ████ 8.2%
5. Temperature (temperature_mean)    ███ 5.4%
6. Humidity / Wind / Location        █ 4.0%
```

## 6. Model Limitations & Risks
- **Data Sparsity**: Districts with < 10 historical records have higher variance predictions.
- **Weather API Dependency**: Requires accurate seasonal weather aggregates from Open-Meteo.
- **Leakage Prevention**: Cultivated area (`area`) is included as a log-transformed scale factor; predictions reflect yield intensity per unit area.
- **Outlier Distortion**: Commercial crops (e.g., Coconuts measured in 'Nuts') massively inflate MSE/MAE. Ex-coconut R² is >0.85, whereas global R² drops due to scale mismatch.

## 7. Assumptions & Business Impact
- **Assumption**: The model assumes that state-reported agricultural area remains constant throughout the harvest season without mid-season crop destruction.
- **Assumption**: Historical weather data from Open-Meteo is a sufficient proxy for hyper-local microclimate events (which may not always be true).
- **Business Impact**: Provides government planners, agri-tech startups, and financial institutions with a baseline yield estimation mechanism for risk assessment and supply chain planning.

## 8. Ethical Considerations
- **Fairness**: The model may underperform in historically marginalized or smaller agricultural districts where data collection is sparse. This could lead to under-allocation of resources if used for predictive funding.
- **Transparency**: SHAP explainability is provided via the Streamlit interface to ensure farmers and policymakers understand exactly *why* a yield was predicted.
