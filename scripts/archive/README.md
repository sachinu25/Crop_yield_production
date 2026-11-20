# Archived scripts

These are earlier experiment iterations (CatBoost v1–v4, "grandmaster" tuning runs, ad-hoc checks).
They are kept for history only and are **not** part of the current pipeline.

Current pipeline: `scripts/train_lightgbm.py` (training, benchmarking, artifacts) and `src/` (app + inference).
`train_final_catboost.py` produced `models/CatBoost_best_final.cbm` and `reports/catboost_predictions.csv`,
which are still used as the CatBoost benchmark.
