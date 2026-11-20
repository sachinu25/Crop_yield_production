"""
Prediction service tests against the real LightGBM artifacts in models/.

Skipped automatically if the model artifacts or lookup table are missing.
"""

import json

import lightgbm  # noqa: F401  (import before pandas, see prediction_service)
import numpy as np
import pytest

from src.config.settings import PROJECT_ROOT, settings
from src.utils.exceptions import PredictionError

ARTIFACTS = [settings.MODEL_PATH, PROJECT_ROOT / "models" / "lightgbm_artifacts.json",
             PROJECT_ROOT / "models" / "inference_lookup.parquet", PROJECT_ROOT / "models" / "pca_info.json"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="model artifacts not present")


@pytest.fixture
def sample_input():
    return {
        "crop_year": 2021,
        "state_name": "punjab",
        "district_name": "ludhiana",
        "season": "rabi",
        "crop": ["wheat"],
        "area": 100.0,
        "temperature_2m_mean": 18.0,
        "precipitation_sum": 80.0,
        "relative_humidity_2m_mean": 60.0,
        "wind_speed_10m_mean": 8.0,
        "latitude": 30.9,
        "longitude": 75.85,
    }


def test_predict_yield_returns_plausible_wheat_yield(sample_input):
    from src.services.prediction_service import predict_yield
    result = predict_yield(sample_input)
    assert set(result) == {"wheat"}
    assert 1.0 < result["wheat"] < 10.0  # Punjab wheat is typically ~4-5 t/ha


def test_predict_yield_list_shap_values_sum_to_model_output(sample_input):
    from src.services import prediction_service as ps
    out = ps.predict_yield_list(sample_input)[0]
    assert out["Crop"] == "wheat" and out["Yield"] > 0
    assert len(out["ShapValues"]) == len(out["FeatureNames"]) == 60
    # TreeSHAP is additive: base + sum(contributions) == raw log-space prediction
    raw_log = out["ShapBaseValue"] + sum(out["ShapValues"])
    artifacts = json.loads((PROJECT_ROOT / "models" / "lightgbm_artifacts.json").read_text())
    expected = (np.expm1(raw_log) + 1.0) * artifacts["smearing"].get("wheat", 1.0) - 1.0
    assert out["Yield"] == pytest.approx(expected, abs=1e-3)


def test_unknown_state_raises_prediction_error(sample_input):
    from src.services.prediction_service import predict_yield
    sample_input["state_name"] = "atlantis"
    with pytest.raises(PredictionError):
        predict_yield(sample_input)
