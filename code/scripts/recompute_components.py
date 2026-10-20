"""Recompute the component analysis from saved outputs; no fitting."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
df=pd.read_csv(sys.argv[1]).sort_values("row_index").reset_index(drop=True)
if len(df)!=16581 or not np.array_equal(df.row_index,np.arange(16581)):
    raise ValueError("Expected all 16581 unique principal rows")

def boot(v, seed=20260822, reps=2000):
    cells = df.outer_cell.to_numpy(); uniq = np.unique(cells); pos = {c: np.where(cells == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = [v[np.concatenate([pos[c] for c in rng.choice(uniq, len(uniq), replace=True)])].mean() for _ in range(reps)]
    return [float(q) for q in np.percentile(vals, [2.5, 97.5])]


res = {"table": {}, "gate_effect": {}, "scale_effect": {}}
for m in ("full", "partial"):
    for s in ("unit", "partial"):
        for g in ("nogate", "gate"):
            k = f"{m}_{s}_{g}"
            res["table"][k] = {"adopt": float(df[f"{k}__commit"].mean() * 100), "labels": int(df[f"{k}__lab"].sum()),
                               "c2i": int(df[f"{k}__c2i"].sum()), "delta_rmse": float(df[f"{k}__d"].mean()),
                               "events": int(df[f"{k}__tail"].sum()), "union": int(df[f"{k}__union"].sum()),
                               "f1": float(f1_score(df.label, df[f"{k}__pred"], labels=np.arange(5), average="macro", zero_division=0))}
        a, b = f"{m}_unit_gate", f"{m}_unit_nogate"
        res["gate_effect"][m] = {"rmse": [float((df[f"{a}__d"] - df[f"{b}__d"]).mean()), boot((df[f"{a}__d"] - df[f"{b}__d"]).to_numpy())],
                                 "event_rate": [float((df[f"{a}__tail"] - df[f"{b}__tail"]).mean()), boot((df[f"{a}__tail"] - df[f"{b}__tail"]).to_numpy(float))]}
        a, b = f"{m}_partial_gate", f"{m}_unit_gate"
        res["scale_effect"][m] = {"rmse": [float((df[f"{a}__d"] - df[f"{b}__d"]).mean()), boot((df[f"{a}__d"] - df[f"{b}__d"]).to_numpy())],
                                  "event_rate": [float((df[f"{a}__tail"] - df[f"{b}__tail"]).mean()), boot((df[f"{a}__tail"] - df[f"{b}__tail"]).to_numpy(float))]}
json.dump(res, open(sys.argv[2], "w"), indent=1)
for k, v in res["table"].items():
    print(f"{k:24s} adopt {v['adopt']:6.2f} labels {v['labels']:5d} c2i {v['c2i']:5d} dRMSE {v['delta_rmse']:+.4f} events {v['events']:5d} union {v['union']:5d} F1 {v['f1']:.4f}")
print(json.dumps({k: res[k] for k in ("gate_effect", "scale_effect")}, indent=1))
