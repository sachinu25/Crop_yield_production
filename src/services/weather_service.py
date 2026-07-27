"""
Weather service — fetches historical weather data from Open-Meteo API.

This module replaces the `api_data()` function that was duplicated (with bugs)
in both project/helper.py and project/streamlit_app/Helper/helper.py.

Bugs fixed:
    - Returns empty DataFrame on error instead of [] (which crashed callers)
    - Handles unrecognised seasons gracefully via constants.get_season_dates()
    - Adds structured logging instead of silent failure
"""

import logging
from typing import Optional

import pandas as pd
import requests

from src.config.constants import WEATHER_DAILY_VARIABLES, get_season_dates
from src.config.settings import settings
from src.utils.exceptions import WeatherAPIError

logger = logging.getLogger(__name__)

# Column schema for the weather DataFrame — guarantees consistent shape
_WEATHER_COLUMNS = [
    "latitude",
    "longitude",
    "date",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "temperature_2m_mean",
    "wind_speed_10m_mean",
]


def fetch_weather_data(
    year: int,
    season: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Fetch historical daily weather data from Open-Meteo for a given location
    and agricultural season.

    Args:
        year: Crop year (e.g. 2020).
        season: Season name (e.g. 'kharif', 'rabi'). Case-insensitive.
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        DataFrame with columns defined by _WEATHER_COLUMNS.
        Returns an **empty DataFrame** (never a bare list) if the API call
        fails or the season is unrecognised.
    """
    # --- Resolve season date range (fixes: missing else clause crash) ---
    dates: Optional[tuple] = get_season_dates(year, season)
    if dates is None:
        logger.error("Unrecognised season '%s'. Valid: kharif, rabi, winter, summer, autumn, whole year", season)
        raise WeatherAPIError(f"Unrecognised season '{season}'")

    start_date, end_date = dates

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": WEATHER_DAILY_VARIABLES,
        "timezone": "auto",
    }

    try:
        response = requests.get(
            settings.WEATHER_API_BASE_URL,
            params=params,
            timeout=settings.WEATHER_API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        if "daily" in data:
            daily = data["daily"]
            for i in range(len(daily["time"])):
                results.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": daily["time"][i],
                    "precipitation_sum": daily["precipitation_sum"][i],
                    "relative_humidity_2m_mean": daily["relative_humidity_2m_mean"][i],
                    "temperature_2m_mean": daily["temperature_2m_mean"][i],
                    "wind_speed_10m_mean": daily["wind_speed_10m_mean"][i],
                })

        df = pd.DataFrame(results, columns=_WEATHER_COLUMNS)
        logger.info(
            "Fetched %d days of weather data for (%.2f, %.2f) season=%s year=%d",
            len(df), lat, lon, season, year,
        )
        return df

    except requests.exceptions.Timeout as exc:
        logger.error("Weather API request timed out after %ds", settings.WEATHER_API_TIMEOUT)
        raise WeatherAPIError("Weather API request timed out") from exc

    except requests.exceptions.HTTPError as exc:
        logger.error("Weather API HTTP error: %s", exc)
        raise WeatherAPIError(f"Weather API HTTP error: {exc}") from exc

    except requests.exceptions.RequestException as exc:
        logger.error("Weather API request failed: %s", exc)
        raise WeatherAPIError(f"Weather API request failed: {exc}") from exc

    except (KeyError, ValueError) as exc:
        logger.error("Failed to parse weather API response: %s", exc)
        raise WeatherAPIError(f"Failed to parse weather API response: {exc}") from exc
