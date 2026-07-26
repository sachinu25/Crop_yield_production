from unittest.mock import patch
import pytest
import pandas as pd
import requests

from src.services.weather_service import fetch_weather_data
from src.utils.exceptions import WeatherAPIError


def test_fetch_weather_data_success(mock_weather_response):
    """Test successful weather data fetch and DataFrame creation."""
    with patch("src.services.weather_service.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_weather_response

        df = fetch_weather_data(year=2020, season="kharif", lat=31.63, lon=74.87)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "precipitation_sum" in df.columns
        assert df.iloc[0]["precipitation_sum"] == 5.0


def test_fetch_weather_data_invalid_season():
    """Test fetching weather for an unrecognised season."""
    with pytest.raises(WeatherAPIError, match="Unrecognised season"):
        fetch_weather_data(year=2020, season="invalid_season", lat=31.63, lon=74.87)


def test_fetch_weather_data_timeout():
    """Test handling of API timeout."""
    with patch("src.services.weather_service.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        
        with pytest.raises(WeatherAPIError, match="timed out"):
            fetch_weather_data(year=2020, season="kharif", lat=31.63, lon=74.87)


def test_fetch_weather_data_http_error():
    """Test handling of HTTP errors (e.g., 404, 500)."""
    with patch("src.services.weather_service.requests.get") as mock_get:
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        with pytest.raises(WeatherAPIError, match="HTTP error"):
            fetch_weather_data(year=2020, season="kharif", lat=31.63, lon=74.87)
