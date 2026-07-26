"""
Land-area unit conversion utilities.

Replaces the two functions `unit_conversion()` and `conversion_factor_Ha_to_X()`
from the old Streamlit Helper/helper.py, which contained an **identical** 22-line
dictionary duplicated across both functions (DRY violation).

Now uses a single dictionary from `src.config.constants`.
"""

from src.config.constants import AREA_CONVERSION_TO_HA


def convert_to_hectares(value: float, unit: str) -> float:
    """
    Convert a land area from the given unit to hectares.

    Args:
        value: The area measurement.
        unit: The unit of measurement (e.g. 'acre', 'bigha', 'sq_km').

    Returns:
        Area in hectares.

    Raises:
        KeyError: If the unit is not recognised.
    """
    if unit not in AREA_CONVERSION_TO_HA:
        raise KeyError(
            f"Unknown area unit '{unit}'. "
            f"Supported: {', '.join(AREA_CONVERSION_TO_HA.keys())}"
        )
    return value * AREA_CONVERSION_TO_HA[unit]


def hectares_per_unit(unit: str) -> float:
    """
    Return how many hectares are in one unit of the given measurement.

    This is used to convert yield (tonnes/ha) into yield per user-chosen unit.

    Args:
        unit: The unit of measurement.

    Returns:
        Number of hectares per one unit of the given measurement.

    Raises:
        KeyError: If the unit is not recognised.
    """
    if unit not in AREA_CONVERSION_TO_HA:
        raise KeyError(
            f"Unknown area unit '{unit}'. "
            f"Supported: {', '.join(AREA_CONVERSION_TO_HA.keys())}"
        )
    return AREA_CONVERSION_TO_HA[unit]
