"""
Application constants.

Single source of truth for season date ranges, weather API parameters,
and land-area conversion factors. Previously these were hardcoded in
multiple helper files and duplicated between Flask and Streamlit.
"""

from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Indian agricultural season date ranges
# ---------------------------------------------------------------------------
# Maps season name → (start_month-day, end_month-day, crosses_year_boundary)
SEASON_DATE_RANGES: Dict[str, Tuple[str, str, bool]] = {
    "kharif":     ("06-01", "08-31", False),
    "rabi":       ("02-01", "03-31", False),
    "winter":     ("11-01", "01-31", True),   # Nov → Jan next year
    "summer":     ("04-01", "05-31", False),
    "autumn":     ("09-01", "10-31", False),
    "whole year": ("02-01", "01-31", True),   # Feb → Jan next year
}

# ---------------------------------------------------------------------------
# Weather API daily variables to request
# ---------------------------------------------------------------------------
WEATHER_DAILY_VARIABLES: str = (
    "precipitation_sum,"
    "relative_humidity_2m_mean,"
    "temperature_2m_mean,"
    "wind_speed_10m_mean"
)

# ---------------------------------------------------------------------------
# CatBoost categorical feature names (used during inference)
# ---------------------------------------------------------------------------
CATBOOST_CAT_FEATURES: list = ["season", "crop"]

# ---------------------------------------------------------------------------
# Land-area conversion factors (to hectares)
#
# Multiply a value in the given unit by this factor to get hectares.
# Single source of truth — previously duplicated in two functions.
# ---------------------------------------------------------------------------
AREA_CONVERSION_TO_HA: Dict[str, float] = {
    "ha":       1.0,
    "sq_m":     1 / 10_000,
    "sq_km":    100.0,
    "acre":     0.4046856422,
    "sq_ft":    0.0000092903,
    "sq_yd":    0.0000836127,
    "gaj":      0.0000836,
    "kanal":    0.0505857,
    "bigha":    0.1011714,
    "biswa":    0.0050586,
    "killa":    0.4046856,
    "lessa":    0.0002529,
    "dhur":     0.0002529,
    "pura":     0.4046856,
    "chatak":   0.0004181,
    "marla":    0.0025293,
    "katha":    0.0050586,
    "ground":   0.0223000,
    "cent":     0.0040465,
    "murabba":  10.1171411,
    "guntha":   0.0101171,
    "karam":    0.0002810,
}

# Supported unit names (useful for dropdowns)
SUPPORTED_AREA_UNITS: list = list(AREA_CONVERSION_TO_HA.keys())


def get_season_dates(year: int, season: str) -> Optional[Tuple[str, str]]:
    """
    Return (start_date, end_date) ISO strings for a given crop year and season.

    Args:
        year: The crop year (e.g. 2020).
        season: Season name (lowercase). Must be a key in SEASON_DATE_RANGES.

    Returns:
        Tuple of (start_date, end_date) as 'YYYY-MM-DD' strings,
        or None if the season is not recognised.
    """
    season_lower = season.lower().strip()
    if season_lower not in SEASON_DATE_RANGES:
        return None

    start_md, end_md, crosses_year = SEASON_DATE_RANGES[season_lower]

    start_date = f"{year}-{start_md}"
    end_year = year + 1 if crosses_year else year
    end_date = f"{end_year}-{end_md}"

    return start_date, end_date
