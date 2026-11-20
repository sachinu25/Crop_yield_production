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

def test_pred(modifications):
    inp = base_input.copy()
    inp.update(modifications)
    res = predict_yield_list(inp)
    print(f"Mod: {modifications} -> Yield: {res[0]['Yield']}")

test_pred({})
test_pred({"crop": ["wheat"], "season": "rabi"})
test_pred({"crop": ["sugarcane"], "season": "kharif"})
test_pred({"state_name": "punjab", "district_name": "ludhiana"})
test_pred({"temperature_2m_mean": 35.0, "precipitation_sum": 100.0})
test_pred({"crop_year": 2030})
