import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.services.prediction_service import predict_yield_list

pred_input = {
    "crop_year": 2026,
    "season": "kharif     ",
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

try:
    res = predict_yield_list(pred_input)
    print("SUCCESS:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
