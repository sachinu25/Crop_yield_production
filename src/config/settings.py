"""
Centralized application settings.

Loads configuration from environment variables and .env file,
replacing all previously hardcoded values throughout the project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve project root (parent of src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Load .env from project root (if it exists)
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Application-wide configuration loaded from environment variables."""

    # -- Flask --
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # -- Weather API --
    WEATHER_API_BASE_URL: str = os.getenv(
        "WEATHER_API_BASE_URL",
        "https://archive-api.open-meteo.com/v1/archive",
    )
    WEATHER_API_TIMEOUT: int = int(os.getenv("WEATHER_API_TIMEOUT", "30"))

    # -- Data Paths --
    DATA_CSV_PATH: Path = PROJECT_ROOT / os.getenv(
        "DATA_CSV_PATH", "data/processed/final_data_for_web.csv"
    )
    MODEL_PATH: Path = PROJECT_ROOT / os.getenv(
        "MODEL_PATH", "models/CatBoost_best_final.cbm"
    )

    # -- Logging --
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # -- MySQL Database --
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB: str = os.getenv("MYSQL_DB", "indian_agriculture")


# Module-level singleton so every import gets the same instance
settings = Settings()
