"""
Data service — loads and queries the agriculture dataset.

This module replaces the scattered CSV-loading logic that was duplicated
in app.py (Flask) and Helper/helper.py (Streamlit).

Improvements:
    - Dataset loaded once and cached (not on every request)
    - Dropdown options computed once and cached
    - Lat/lon extraction with proper error handling
    - Type hints and docstrings throughout
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import settings
from src.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Module-level cache for the DataFrame
_data_df: Optional[pd.DataFrame] = None


def _load_data() -> pd.DataFrame:
    """Load and cache the web dataset from the configured CSV path."""
    global _data_df
    if _data_df is None:
        csv_path = str(settings.DATA_CSV_PATH)
        logger.info("Loading dataset from %s", csv_path)
        _data_df = pd.read_csv(csv_path)
        logger.info("Dataset loaded: %d rows, %d columns", *_data_df.shape)
    return _data_df


def get_dataframe() -> pd.DataFrame:
    """Return the cached agriculture DataFrame."""
    return _load_data()


def get_dropdown_options(title_case: bool = False) -> Dict[str, Any]:
    """
    Compute dropdown options for the UI (states, districts, seasons, crops).

    Args:
        title_case: If True, convert state/season/district names to Title Case.
                    Used by the Streamlit app (which displays Title Case).
                    Flask app uses raw lowercase values.

    Returns:
        Dict with keys: state_list, district_dict, season_list, crop_list.
    """
    df = _load_data()

    state_list: List[str] = sorted(df["state_name"].dropna().unique().tolist())
    district_dict: Dict[str, List[str]] = {
        state: sorted(df[df["state_name"] == state]["district_name"].dropna().unique().tolist())
        for state in state_list
    }
    season_list: List[str] = sorted(df["season"].dropna().unique().tolist())
    crop_list: List[str] = sorted(df["crop"].dropna().unique().tolist())

    if title_case:
        season_list = [s.title() for s in season_list]
        state_list_tc = [s.title() for s in state_list]
        district_dict_tc = {
            s.title(): [d.title() for d in districts]
            for s, districts in district_dict.items()
        }
        return {
            "state_list": state_list_tc,
            "district_dict": district_dict_tc,
            "season_list": season_list,
            "crop_list": crop_list,
        }

    return {
        "state_list": state_list,
        "district_dict": district_dict,
        "season_list": season_list,
        "crop_list": crop_list,
    }


def get_lat_lon(state: str, district: str) -> Tuple[float, float]:
    """
    Extract latitude and longitude for a given state + district.

    Args:
        state: State name (matched case-insensitively).
        district: District name (matched case-insensitively).

    Returns:
        Tuple of (latitude, longitude).

    Raises:
        ValueError: If the state/district combination is not found.
    """
    df = _load_data()
    location_row = df[
        (df["state_name"] == state.lower()) &
        (df["district_name"] == district.lower())
    ]

    if location_row.empty:
        raise DataValidationError(
            f"Location not found: state='{state}', district='{district}'"
        )

    lat = float(location_row["latitude"].values[0])
    lon = float(location_row["longitude"].values[0])
    return lat, lon
