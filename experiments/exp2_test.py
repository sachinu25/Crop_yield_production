"""Experiment 2: refit the dev-selected options on 2001-2017 and score ONCE on the 2018-2020 test set.
Also diagnose why test R2 (~0.60) is so much lower than dev R2 (~0.87)."""
import json, warnings
import numpy as np
import lightgbm as lgb  # imported before pandas (avoids a DLL load-order crash on some Windows setups)
import pandas as pd
from common import load, to_lgb, cat_levels, baseline, metrics

warnings.filterwarnings("ignore")
Xtr, ytr, Xte, yte = load()
lv = cat_levels(Xtr)
Ltr, Lte = to_lgb(Xtr, lv), to_lgb(Xte, lv)
bte = baseline(Xte)
P = dict(learning_rate=0.03, num_leaves=255, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_cat_to_onehot=8,
         cat_smooth=20, verbose=-1, seed=42, n_jobs=-1)
# iteration counts picked on the dev split (exp1), scaled ~1.15x for the larger training set
ITERS = {"log": 140, "raw": 120, "tweedie": 125}


def train(y, objective, n):
    return lgb.train({**P, "objective": objective, **({"tweedie_variance_power": 1.5} if objective == "tweedie" else {})},
                     lgb.Dataset(Ltr, label=np.ascontiguousarray(y, dtype=np.float64), free_raw_data=False), num_boost_round=n)


res = {"baseline_roll3_district_crop": metrics(yte, bte, Xte["crop"])}
mA = train(np.log1p(ytr), "regression", ITERS["log"])
pA = np.expm1(mA.predict(Lte))
res["A_log_target (current approach)"] = metrics(yte, pA, Xte["crop"])
sm = pd.Series(np.exp(np.log1p(ytr) - mA.predict(Ltr))).groupby(Xtr["crop"].values).mean()
pB = (pA + 1) * Xte["crop"].map(sm).fillna(1.0).to_numpy() - 1
res["B_log_smearing"] = metrics(yte, pB, Xte["crop"])
pC = train(ytr, "regression", ITERS["raw"]).predict(Lte)
res["C_raw_target"] = metrics(yte, pC, Xte["crop"])
pE = train(ytr, "tweedie", ITERS["tweedie"]).predict(Lte)
res["E_tweedie"] = metrics(yte, pE, Xte["crop"])
res["F_blend_B_C_E"] = metrics(yte, (pB + pC + pE) / 3, Xte["crop"])

# ---- diagnosis: are a few impossible test values dominating the error? ----
d = pd.DataFrame({"crop": Xte["crop"].values, "year": Xte["crop_year"].values, "y": yte, "p": pB})
train_max = pd.Series(ytr).groupby(Xtr["crop"].values).max()
d["train_max"] = d["crop"].map(train_max)
d["impossible"] = d["y"] > 2 * d["train_max"]          # >2x the highest yield ever seen for that crop in 17 years
d["sse"] = (d.y - d.p) ** 2
tot = d["sse"].sum()
top = d.sort_values("sse", ascending=False)
diag = {
    "test_rows": len(d),
    "rows_above_2x_train_max": int(d["impossible"].sum()),
    "share_of_SSE_from_those_rows_%": round(100 * d.loc[d.impossible, "sse"].sum() / tot, 1),
    "share_of_SSE_from_top_50_rows_%": round(100 * top["sse"].head(50).sum() / tot, 1),
    "top_10_errors": top.head(10)[["crop", "year", "y", "p", "train_max"]].round(1).to_dict("records"),
}
clean = ~d["impossible"].to_numpy()
res["B_log_smearing | test without >2x-impossible rows"] = metrics(yte[clean], pB[clean], Xte["crop"][clean])
res["baseline | test without >2x-impossible rows"] = metrics(yte[clean], bte[clean], Xte["crop"][clean])
res["C_raw | test without >2x-impossible rows"] = metrics(yte[clean], pC[clean], Xte["crop"][clean])
print(json.dumps({"results": res, "diagnosis": diag}, indent=1, default=str))

