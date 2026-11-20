"""
Single source of truth for model metrics shown in the app.

Reads models/evaluation.json, written by scripts/train_lightgbm.py from the
2018-2020 hold-out predictions. Returns None if the file is missing so pages
can degrade gracefully instead of showing made-up numbers.
"""

from __future__ import annotations

import json
from functools import lru_cache

from src.config.settings import PROJECT_ROOT

EVALUATION_PATH = PROJECT_ROOT / "models" / "evaluation.json"


@lru_cache(maxsize=1)
def load_evaluation() -> dict | None:
    try:
        with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def headline_metrics() -> dict | None:
    """Compact numbers for cards: production model vs naive baseline, full and quality-filtered."""
    ev = load_evaluation()
    if not ev:
        return None
    prod = ev["benchmark"][ev["production_model"]]
    base = ev["benchmark"]["Naive baseline"]
    return {
        "model_name": ev["production_model"],
        "r2_clean": prod["quality_filtered"]["r2"],
        "mae_clean": prod["quality_filtered"]["mae"],
        "rmse_clean": prod["quality_filtered"]["rmse"],
        "r2_full": prod["full"]["r2"],
        "mae_full": prod["full"]["mae"],
        "baseline_r2_clean": base["quality_filtered"]["r2"],
        "mae_gain_vs_baseline_pct": ev["mae_reduction_vs_baseline_pct"],
        "n_train": ev["split"]["n_train"],
        "n_test": ev["split"]["n_test"],
        "n_features": ev["split"]["n_features"],
        "train_years": ev["split"]["train_years"],
        "test_years": ev["split"]["test_years"],
        "flagged_rows": ev["error_analysis"]["flagged_rows"],
        "flagged_pct": ev["error_analysis"]["flagged_pct_of_test"],
        "flagged_err_share": ev["error_analysis"]["flagged_share_of_squared_error_pct"],
    }
