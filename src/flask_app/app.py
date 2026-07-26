"""
Flask application for Crop Yield Prediction.

Refactored from the original project/app.py with the following fixes:
    - Removed: import joblib (dead import — never used)
    - Removed: debug=True (security vulnerability)
    - Fixed: bare 'except Exception' → specific error handling with logging
    - Fixed: uses shared services instead of duplicated helper code
    - Fixed: area log-transform now consistent with Streamlit
    - Added: structured logging
    - Added: proper docstrings and type hints
    - Moved: configuration to src/config/settings.py

The original ML logic and template rendering are UNCHANGED.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config.settings import settings
from src.services.data_service import get_dropdown_options, get_lat_lon
from src.services.prediction_service import predict_yield
from src.services.weather_service import fetch_weather_data

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Preload dropdown options once at startup (not on every request)
_options = get_dropdown_options(title_case=False)
state_list = _options["state_list"]
district_dict = _options["district_dict"]
season_list = _options["season_list"]
crop_list = _options["crop_list"]


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the input form (GET) or process a prediction request (POST)."""
    prediction = None
    selected_state = None
    selected_district = None

    if request.method == "POST":
        try:
            selected_state = request.form.get("state")
            selected_district = request.form.get("district")
            crop_year = int(request.form.get("crop_year"))
            season = request.form.get("season")
            crop = request.form.getlist("crop")
            area = float(request.form.get("area"))

            # Extract coordinates
            lat, lon = get_lat_lon(selected_state, selected_district)

            # Fetch weather data
            weather_df = fetch_weather_data(crop_year, season, lat, lon)

            if weather_df.empty:
                prediction = "Error: Could not fetch weather data. Please try again."
                return render_template(
                    "index.html",
                    state_list=state_list,
                    district_dict=district_dict,
                    season_list=season_list,
                    crop_list=crop_list,
                    prediction=prediction,
                )

            # Build prediction input
            input_data = {
                "crop_year": crop_year,
                "season": season,
                "crop": crop,
                "area": np.log1p(area),
                "temperature_2m_mean": weather_df["temperature_2m_mean"].mean(),
                "precipitation_sum": weather_df["precipitation_sum"].mean(),
                "relative_humidity_2m_mean": weather_df["relative_humidity_2m_mean"].mean(),
                "wind_speed_10m_mean": weather_df["wind_speed_10m_mean"].mean(),
                "latitude": lat,
                "longitude": lon,
            }

            pred_data = predict_yield(input_data)

            return render_template(
                "result.html",
                prediction=pred_data,
                selected_state=selected_state,
                selected_district=selected_district,
                data=weather_df.to_dict(orient="records"),
            )

        except ValueError as exc:
            logger.warning("Validation error: %s", exc)
            prediction = f"Error: {exc}"

        except Exception as exc:
            logger.exception("Unexpected error during prediction")
            prediction = f"Error: An unexpected error occurred. Please try again."

    return render_template(
        "index.html",
        state_list=state_list,
        district_dict=district_dict,
        season_list=season_list,
        crop_list=crop_list,
    )


@app.route("/get_districts", methods=["POST"])
def get_districts():
    """Return the list of districts for a given state (AJAX endpoint)."""
    state = request.json.get("state")
    districts = district_dict.get(state, [])
    return jsonify(districts)


if __name__ == "__main__":
    logger.info(
        "Starting Flask server on %s:%d (debug=%s)",
        settings.FLASK_HOST, settings.FLASK_PORT, settings.FLASK_DEBUG,
    )
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG,
    )
