"""Experiment 1: choose a target/objective on a DEV split only (train <=2014, validate 2015-2017).
The real test years (2018-2020) are not touched here."""
import json, time, warnings
import numpy as np
import lightgbm as lgb
from common import load, to_lgb, cat_levels, baseline, metrics

warnings.filterwarnings("ignore")
Xall, yall, _, _ = load()
tr = Xall["crop_year"] <= 2014
va = Xall["crop_year"] >= 2015
Xtr, Xva, ytr, yva = Xall[tr], Xall[va], yall[tr.values], yall[va.values]
lv = cat_levels(Xtr)
Ltr, Lva = to_lgb(Xtr, lv), to_lgb(Xva, lv)
btr, bva = baseline(Xtr), baseline(Xva)

P = dict(learning_rate=0.03, num_leaves=255, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_cat_to_onehot=8,
         cat_smooth=20, verbose=-1, seed=42, n_jobs=-1)


def fit(target_tr, target_va, objective="regression", extra=None):
    params = {**P, "objective": objective, **(extra or {})}
    dtr = lgb.Dataset(Ltr, target_tr)
    dva = lgb.Dataset(Lva, target_va, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


res = {"baseline_roll3_district_crop": metrics(yva, bva, Xva["crop"])}
t = time.time()

# A: current approach -- log1p target
m = fit(np.log1p(ytr), np.log1p(yva))
pA = np.expm1(m.predict(Lva, num_iteration=m.best_iteration))
res["A_log_target"] = {**metrics(yva, pA, Xva["crop"]), "iters": m.best_iteration}

# B: log target + Duan smearing correction per crop (fixes expm1 under-prediction of the mean)
resid_tr = np.log1p(ytr) - m.predict(Ltr, num_iteration=m.best_iteration)
import pandas as pd
sm = pd.Series(np.exp(resid_tr)).groupby(Xtr["crop"].values).mean()
pB = (pA + 1) * Xva["crop"].map(sm).fillna(1.0).to_numpy() - 1
res["B_log_smearing"] = metrics(yva, pB, Xva["crop"])

# C: raw target, L2
m = fit(ytr, yva)
pC = m.predict(Lva, num_iteration=m.best_iteration)
res["C_raw_target"] = {**metrics(yva, pC, Xva["crop"]), "iters": m.best_iteration}

# D: log-ratio to the naive baseline (model learns the deviation from district history)
m = fit(np.log1p(ytr) - np.log1p(btr), np.log1p(yva) - np.log1p(bva))
pD = np.expm1(np.log1p(bva) + m.predict(Lva, num_iteration=m.best_iteration))
res["D_log_ratio_to_baseline"] = {**metrics(yva, pD, Xva["crop"]), "iters": m.best_iteration}

# E: Tweedie objective on raw target (models the mean of a skewed positive target directly)
m = fit(ytr, yva, objective="tweedie", extra={"tweedie_variance_power": 1.5})
pE = m.predict(Lva, num_iteration=m.best_iteration)
res["E_tweedie_raw"] = {**metrics(yva, pE, Xva["crop"]), "iters": m.best_iteration}

# F: simple blends
res["F_blend_C_E"] = metrics(yva, (pC + pE) / 2, Xva["crop"])
res["F_blend_B_C_E"] = metrics(yva, (pB + pC + pE) / 3, Xva["crop"])

print(json.dumps(res, indent=1))
print(f"done in {time.time()-t:.0f}s")
