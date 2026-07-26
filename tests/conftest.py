import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def mock_weather_response():
    """Mock successful response from Open-Meteo API."""
    return {
        "daily": {
            "time": ["2020-06-01", "2020-06-02"],
            "precipitation_sum": [5.0, 12.0],
            "relative_humidity_2m_mean": [65.0, 70.0],
            "temperature_2m_mean": [28.0, 27.5],
            "wind_speed_10m_mean": [12.0, 15.0]
        }
    }

@pytest.fixture
def mock_invalid_weather_response():
    """Mock invalid response from Open-Meteo API."""
    return {"error": True, "reason": "Invalid parameters"}

@pytest.fixture
def sample_raw_dataframe():
    """Sample raw agriculture dataset for ETL validation tests."""
    return pd.DataFrame({
        "state_name": ["Punjab", "Punjab", None],
        "district_name": ["Amritsar", "Ludhiana", "Unknown"],
        "season": ["Kharif", "Rabi", "Kharif"],
        "crop": ["Rice", "Wheat", "Maize"],
        "crop_year": [2020, 2020, 2020],
        "area": [1000.5, -500.0, 200.0],
        "crop_yield": [4.2, 3.8, 1.5],
        "temperature_2m_mean": [32.0, 18.0, 25.0],
        "precipitation_sum": [500.0, 50.0, 100.0]
    })
