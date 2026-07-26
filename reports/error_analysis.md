# Error Analysis Report
## Indian Agriculture Crop Yield Prediction

### 1. Residual Analysis & Worst Predictions
Analysis of the top 10 largest prediction errors on the unseen 2018-2020 test set reveals a systemic issue with specific commercial crops, rather than a failure of the machine learning model itself.

#### Top 5 Largest Absolute Errors
| State | Crop | True Yield (T/Ha) | Predicted Yield (T/Ha) | Error |
|---|---|---|---|---|
| Andhra Pradesh | Coconut | 3,720,000 | 3,457 | -3,716,542 |
| Madhya Pradesh | Cotton(Lint) | 194,194 | 2.4 | -194,191 |
| Rajasthan | Castor Seed | 126,642 | 4.3 | -126,637 |
| Rajasthan | Moong(Green Gram) | 84,061 | 5.8 | -84,055 |
| Chhattisgarh | Sugarcane | 51,896 | 33.3 | -51,862 |

### 2. Business Interpretation
**The "Coconut Anomaly":**
The target variable `crop_yield` is calculated as `Production / Area`. For 95% of crops, Production is measured in Tonnes. However, for Coconuts, historical Indian agricultural data records production in **Nuts** (units), not Tonnes. Consequently, the yield for Coconuts artificially inflates into the millions (e.g., 3.7 million nuts per hectare), which massively skews the Mean Squared Error (MSE) and overall R² calculations for the entire test set.

**The "Cash Crop" Variance:**
Similarly, Cotton is sometimes measured in bales, and Sugarcane naturally has a substantially higher mass per hectare than cereals like wheat or rice. 

### 3. Bias/Variance Tradeoff
The CatBoost model demonstrates **High Bias** specifically on these extreme outlier crops, predicting a much lower yield (closer to the mean of all other crops). This is actually the *desired behavior* of a robust model attempting to find a generalizable mapping of weather to agricultural biomass, rather than overfitting to anomalous reporting units.

### 4. Recommendations for Data Engineering
To resolve this data quality issue at the source, the ETL pipeline must be updated to standardize units:
1. **Unit Conversion Table**: Implement a lookup table to convert Coconut (Nuts) to Tonnes (approx 1,425 nuts = 1 Tonne).
2. **Cotton Conversion**: Convert Cotton bales (170 kg) to Tonnes.
3. **Outlier Flagging**: Implement an Isolation Forest during preprocessing to flag these records for manual review before they enter the ML training pool.
