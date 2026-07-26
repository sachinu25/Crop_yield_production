"""
Custom exceptions for the Indian Agriculture Crop Yield Prediction project.

These exceptions provide specific error types for different failure modes
in the application, allowing for more granular error handling and logging.
"""

class AgricultureBaseError(Exception):
    """Base exception for all custom errors in the project."""
    pass


class PredictionError(AgricultureBaseError):
    """Raised when the ML model fails to generate a prediction."""
    pass


class WeatherAPIError(AgricultureBaseError):
    """Raised when the Open-Meteo API call fails or returns invalid data."""
    pass


class DatabaseError(AgricultureBaseError):
    """Raised when a MySQL connection or query fails."""
    pass


class DataValidationError(AgricultureBaseError):
    """Raised when input data fails validation checks (e.g. nulls, negative values)."""
    pass


class ConfigurationError(AgricultureBaseError):
    """Raised when application configuration or environment variables are missing/invalid."""
    pass
