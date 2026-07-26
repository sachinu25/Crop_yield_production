from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from src.services.prediction_service import predict_yield, predict_yield_list
from src.utils.exceptions import PredictionError


@pytest.fixture
def mock_prediction_data():
    """Mock input data for prediction tests."""
    return {
        "crop_year": 2020,
        "season": "kharif",
        "crop": ["rice"],
        "area": 100.0,
        "temperature_2m_mean": 28.0,
        "precipitation_sum": 500.0,
        "relative_humidity_2m_mean": 70.0,
        "wind_speed_10m_mean": 12.0,
        "latitude": 31.63,
        "longitude": 74.87
    }


@patch("src.services.prediction_service._load_model")
def test_predict_yield_success(mock_load, mock_prediction_data):
    """Test successful prediction for Flask dictionary format."""
    mock_model = MagicMock()
    # Mocking the log-prediction output. np.expm1(1.0) is approx 1.718
    mock_model.predict.return_value = np.array([1.0])
    mock_load.return_value = mock_model
    
    result = predict_yield(mock_prediction_data)
    
    assert "rice" in result
    assert result["rice"] > 1.7  # np.expm1(1.0) = ~1.718


@patch("src.services.prediction_service._load_model")
def test_predict_yield_list_success(mock_load, mock_prediction_data):
    """Test successful prediction for Streamlit list format."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.0])
    mock_load.return_value = mock_model
    
    result = predict_yield_list(mock_prediction_data)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["Crop"] == "rice"
    assert result[0]["Yield"] > 1.7


@patch("src.services.prediction_service._load_model")
def test_predict_yield_failure(mock_load, mock_prediction_data):
    """Test that underlying model failures raise PredictionError."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("CatBoost internal error")
    mock_load.return_value = mock_model
    
    with pytest.raises(PredictionError, match="Prediction failed"):
        predict_yield(mock_prediction_data)
