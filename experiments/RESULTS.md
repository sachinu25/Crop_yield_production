# Crop yield model: experiment results

Data: the repo's saved checkpoints (train 2001â€“2017, 211,683 rows; test 2018â€“2020, 35,614 rows).
Every model choice was made on a dev split (train 2001â€“2014, validate 2015â€“2017). The 2018â€“2020 test years were scored once at the end.

## Main finding: the RÂ² of 0.60 comes from bad test data, not from the model

- **135 test rows (0.38%)** have yields above **2Ã— the highest yield seen for that crop in 17 training years**.
  Example: guar seed at 485 t/ha, when its training maximum is 3.6 t/ha. These look like data-entry or unit errors.
- Those 135 rows account for **83% of all squared error** on the test set.
- The original pipeline clipped outliers in the training data only. The test data was never cleaned.
- On the full test set, **every model scores RÂ² â‰ˆ 0.60**: the naive baseline, LightGBM, raw/log/Tweedie targets, and the original CatBoost (0.6047).

## Results on 2018â€“2020 (out-of-time)

| Model | Test set | RÂ² | MAE | RÂ² (log) | Median APE |
|---|---|---|---|---|---|
| Naive baseline (last-3-yr district+crop mean) | full | 0.580 | 1.544 | 0.835 | 16.3% |
| Original CatBoost (repo notebook 07) | full | 0.605 | 1.421 | â€“ | â€“ |
| LightGBM, log target + per-crop smearing | full | 0.602 | 1.421 | 0.889 | 17.2% |
| Naive baseline | quality-filtered* | 0.864 | 1.180 | 0.875 | 16.2% |
| **LightGBM, log target + per-crop smearing** | **quality-filtered*** | **0.900** | **1.054** | **0.931** | 17.1% |

\* Drops the 135 rows whose yield is more than 2Ã— that crop's training maximum. The threshold uses training data only.

The model beats the naive baseline by **~11% lower MAE** and lifts RÂ² from 0.86 to 0.90 on quality-checked records.

## What did not help
- Raw target, Tweedie objective, log-ratio-to-baseline target, blends, and a 4-config regularisation sweep. On the dev split all of them landed within Â±0.01 RÂ² of each other.

## Leakage review of the original pipeline: none found
- The lag features use `shift(1)` by year, so they only see earlier years. This is a one-year-ahead forecast.
- Target encodings for the test set come from training statistics only.
- KMeans and PCA are fitted on the training years only.

## Files
- `common.py`: loading, naive baseline, metrics.
- `exp1_dev.py`: target and objective comparison on the dev split.
- `exp2_test.py`: test scoring and outlier diagnosis.
- `exp3_tune.py`: regularisation sweep, final model, quality-filtered evaluation. Saves `lgbm_log_smearing.txt`.

Run from the repo root, for example `python experiments/exp3_tune.py`. It reads the parquet/npy files in `checkpoints/`. Install the extra dependencies with `pip install -r experiments/requirements.txt`.
