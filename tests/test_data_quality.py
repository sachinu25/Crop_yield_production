import numpy as np
import pandas as pd

from src.utils.data_quality import crop_max_yield, impossible_yield_mask, regression_metrics


def test_crop_max_yield_uses_training_values_only():
    mx = crop_max_yield([1.0, 3.5, 2.0, 40.0], ["guar", "guar", "rice", "sugarcane"])
    assert mx.to_dict() == {"guar": 3.5, "rice": 2.0, "sugarcane": 40.0}


def test_impossible_yield_mask_flags_only_values_above_factor():
    train_max = pd.Series({"guar": 3.6, "rice": 5.0})
    y = [485.0, 7.2, 7.3, 9.0, 100.0]
    crops = ["guar", "guar", "guar", "rice", "unseen crop"]
    mask = impossible_yield_mask(y, crops, train_max, factor=2.0)
    # 7.2 == 2 x 3.6 is allowed; unseen crops are never flagged
    assert mask.tolist() == [True, False, True, False, False]


def test_regression_metrics_perfect_and_clipped():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["r2"] == 1.0 and m["mae"] == 0.0 and m["n"] == 3
    # negative predictions are clipped to zero before scoring
    m2 = regression_metrics([1.0, 1.0], [-5.0, 1.0])
    assert m2["mae"] == 0.5
    assert np.isfinite(m2["rmse"])
