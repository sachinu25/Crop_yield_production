"""Experiment 3: small regularisation sweep for the chosen approach (log target + per-crop smearing),
selected on the DEV split only, then scored once on the test years (full and data-quality-filtered)."""
import json, warnings
import numpy as np
import lightgbm as lgb  # imported before pandas (avoids a DLL load-order crash on some Windows setups)
import pandas as pd
from common import load, to_lgb, cat_levels, baseline, metrics

warnings.filterwarnings("ignore")
Xall, yall, Xte, yte = load()
tr, va = (Xall["crop_year"] <= 2014).values, (Xall["crop_year"] >= 2015).values
lv = cat_levels(Xall)
Lall, Lte = to_lgb(Xall, lv), to_lgb(Xte, lv)
Ltr, Lva = Lall[tr], Lall[va]
BASE = dict(objective="regression", bagging_fraction=0.8, bagging_freq=1, max_cat_to_onehot=8,
            verbose=-1, seed=42, n_jobs=-1)
GRID = {
    "current_like": dict(learning_rate=0.03, num_leaves=255, min_child_samples=50, feature_fraction=0.5, lambda_l2=1.0, cat_smooth=20),
    "smaller_trees": dict(learning_rate=0.03, num_leaves=63, min_child_samples=100, feature_fraction=0.5, lambda_l2=5.0, cat_smooth=50),
    "strong_reg": dict(learning_rate=0.02, num_leaves=31, min_child_samples=200, feature_fraction=0.4, lambda_l2=10.0, cat_smooth=100),
    "mid": dict(learning_rate=0.02, num_leaves=127, min_child_samples=150, feature_fraction=0.4, lambda_l2=5.0, cat_smooth=50, min_data_per_group=200),
}


def smear(model, L, y, X):
    return pd.Series(np.exp(np.log1p(y) - model.predict(L))).groupby(X["crop"].values).mean()


def predict(model, L, X, sm):
    return (np.expm1(model.predict(L)) + 1) * X["crop"].map(sm).fillna(1.0).to_numpy() - 1


dev = {}
for name, p in GRID.items():
    m = lgb.train({**BASE, **p}, lgb.Dataset(Ltr, np.log1p(yall[tr])), 5000,
                  valid_sets=[lgb.Dataset(Lva, np.log1p(yall[va]))],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    pv = predict(m, Lva, Xall[va], smear(m, Ltr, yall[tr], Xall[tr]))
    dev[name] = {**metrics(yall[va], pv), "iters": m.best_iteration}
best = max(dev, key=lambda k: dev[k]["R2"])

# refit the dev-selected config on all of 2001-2017 and score once on 2018-2020
n = int(dev[best]["iters"] * 1.15)
m = lgb.train({**BASE, **GRID[best]}, lgb.Dataset(Lall, np.log1p(yall)), n)
pt = predict(m, Lte, Xte, smear(m, Lall, yall, Xall))
bt = baseline(Xte)
train_max = pd.Series(yall).groupby(Xall["crop"].values).max()
clean = ~(yte > 2 * Xte["crop"].map(train_max).to_numpy())   # rule uses training data only
out = {
    "dev_split_2015_2017": dev, "selected": best, "final_iters": n,
    "test_full": {"model": metrics(yte, pt, Xte["crop"]), "baseline": metrics(yte, bt, Xte["crop"])},
    "test_quality_filtered": {"rows_kept": int(clean.sum()), "rows_dropped": int((~clean).sum()),
                              "model": metrics(yte[clean], pt[clean], Xte["crop"][clean]),
                              "baseline": metrics(yte[clean], bt[clean], Xte["crop"][clean])},
}
print(json.dumps(out, indent=1))
m.save_model("lgbm_log_smearing.txt")
