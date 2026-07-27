import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.services.prediction_service import predict_yield_list

base_input = {
    "crop_year": 2026,
    "season": "kharif",
    "crop": ["rice"],
    "area": 1.0,
    "latitude": 20.0,
    "longitude": 80.0,
    "state_name": "maharashtra",
    "district_name": "nagpur",
    "temperature_2m_mean": 28.0,
    "precipitation_sum": 500.0,
    "relative_humidity_2m_mean": 75.0,
    "wind_speed_10m_mean": 3.0
}

res = predict_yield_list(base_input)[0]
print(f"Yield: {res['Yield']}")
print(f"Features (weather): {res['Features']['temperature_2m_mean']}")
print(f"Features (season_sin): {res['Features']['season_sin']}")
print(f"SHAP Values count: {len(res['ShapValues'])}")
print(f"Base Value: {res['ShapBaseValue']}")

# Test fallback logic (missing weather)
fallback_input = base_input.copy()
del fallback_input["temperature_2m_mean"]
del fallback_input["precipitation_sum"]
del fallback_input["relative_humidity_2m_mean"]
del fallback_input["wind_speed_10m_mean"]

res_fallback = predict_yield_list(fallback_input)[0]
print(f"Fallback Yield: {res_fallback['Yield']}")
print(f"Fallback Weather Used: {res_fallback['Features']['temperature_2m_mean']}")
