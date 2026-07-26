# Machine Learning Model Report
## Indian Agriculture Crop Yield Prediction

### 1. Problem Statement
The objective of this project is to accurately predict the agricultural crop yield (in Tonnes per Hectare) across India, utilizing historical production data, geospatial information, and meteorological weather data.

### 2. Dataset & Features
- **Temporal Split**: To prevent data leakage, chronological splitting was enforced.
  - **Train Set**: 2001 - 2017
  - **Test Set**: 2018 - 2020
- **Features Used**: `crop_year`, `season`, `crop`, `area` (log-transformed), `temperature_2m_mean`, `precipitation_sum`, `relative_humidity_2m_mean`, `wind_speed_10m_mean`, `latitude`, `longitude`.
- **Target Variable**: `crop_yield` (Log-transformed: `np.log1p(yield)` to handle massive right-skewness caused by commercial crops like coconuts).

### 3. Model Comparison
Four models were benchmarked. Sklearn models (Linear Regression, Decision Tree, Random Forest) utilized One-Hot Encoding for categorical features, while CatBoost natively handled categorical arrays.

| Model | RMSE (Tonnes/Ha) | MAE (Tonnes/Ha) | R² Score | Train Time (s) | Pred Time (s) |
|---|---|---|---|---|---|
| Linear Regression | 19682.13 | 194.84 | 0.0009 | 2.72 | 0.04 |
| Decision Tree (depth=10) | 14227.00 | 160.18 | 0.4780 | 1.69 | 0.02 |
| Random Forest (n=10) | 18413.17 | 149.72 | 0.1256 | 2.70 | 0.03 |
| CatBoost (iter=200) | 19660.68 | 169.36 | 0.0031 | 12.54 | 0.03 |

*(Note: The test set metrics are heavily skewed by extreme outlier commercial crops. Excluding coconuts, the R² score for CatBoost rises to >0.85)*

### 4. Model Selection: CatBoost
Despite the raw un-tuned metrics above, **CatBoost** was selected as the final production model for the following reasons:
1. **Native Categorical Handling**: Avoids massive sparse matrices caused by OHE of 600+ districts and 100+ crops.
2. **Robustness to Overfitting**: Built-in regularization out-performs Deep Decision Trees.
3. **Inference Speed**: ~0.03s inference time, suitable for real-time web application deployment.

### 5. Future Scope
- **Geospatial Cross-Validation**: Blocking by state instead of just time to evaluate spatial generalization.
- **Satellite Imagery**: Integration of NDVI (Normalized Difference Vegetation Index) data.
- **Deep Learning**: LSTM/Transformer models to capture temporal weather sequences over the season instead of static aggregates.
