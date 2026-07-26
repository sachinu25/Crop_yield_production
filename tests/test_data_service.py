from unittest.mock import patch
import pandas as pd
import pytest

from src.services.data_service import get_dropdown_options, get_lat_lon
from src.utils.exceptions import DataValidationError


@pytest.fixture
def mock_dataset():
    """Mock dataset for data_service tests."""
    return pd.DataFrame({
        "state_name": ["punjab", "punjab", "kerala"],
        "district_name": ["amritsar", "ludhiana", "kollam"],
        "season": ["kharif", "rabi", "kharif"],
        "crop": ["rice", "wheat", "rice"],
        "latitude": [31.63, 30.90, 8.89],
        "longitude": [74.87, 75.85, 76.59]
    })


@patch("src.services.data_service._load_data")
def test_get_dropdown_options(mock_load, mock_dataset):
    mock_load.return_value = mock_dataset
    
    options = get_dropdown_options(title_case=False)
    
    assert "punjab" in options["state_list"]
    assert "kerala" in options["state_list"]
    assert "amritsar" in options["district_dict"]["punjab"]
    assert "rice" in options["crop_list"]


@patch("src.services.data_service._load_data")
def test_get_lat_lon_success(mock_load, mock_dataset):
    mock_load.return_value = mock_dataset
    
    lat, lon = get_lat_lon("punjab", "amritsar")
    
    assert lat == 31.63
    assert lon == 74.87


@patch("src.services.data_service._load_data")
def test_get_lat_lon_not_found(mock_load, mock_dataset):
    mock_load.return_value = mock_dataset
    
    with pytest.raises(DataValidationError, match="Location not found"):
        get_lat_lon("invalid_state", "invalid_district")
