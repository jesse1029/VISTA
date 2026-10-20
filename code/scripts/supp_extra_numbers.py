"""Extra supplement quantities: residual-map contrasts, VISTA-versus-harmonizer contrasts,
grid-edge validation values, and fold-relative envelope exceedances.
Usage: python supp_extra_numbers.py RUN_DIR SPLIT_MANIFEST DESIS OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
run = Path(sys.argv[1]); sm = pd.read_csv(sys.argv[2]).sort_values("row_index").reset_index(drop=True)
base = pd.read_csv(run/"base"/"heldout_per_location.csv").sort_values("row_index").reset_index(drop=True)
ext = pd.read_csv(run/"ext"/"heldout_additional.csv").sort_values("row_index").reset_index(drop=True)
if not np.array_equal(base[["row_index", "loc_id", "fold"]], ext[["row_index", "loc_id", "fold"]]):
    raise ValueError("Principal and additional rows differ in row_index, loc_id, or fold")
if not np.array_equal(base[["row_index", "loc_id"]], sm[["row_index", "loc_id"]]):
    raise ValueError("Split manifest differs from the held-out row order")
df = pd.concat([base, ext.drop(columns=["row_index","loc_id","fold"])], axis=1)
cells = df.outer_cell.to_numpy(); uniq = np.unique(cells); pos = {c: np.where(cells == c)[0] for c in uniq}
def boot(v, seed=20260922, reps=2000):
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        pick = np.concatenate([pos[c] for c in rng.choice(uniq, len(uniq), replace=True)]); out.append(v[pick].mean())
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]
R = {"vs": {}}
for other in ["gated_ridge_partial","gated_pds_partial","gated_pls_partial","gated_krr_partial","gated_mlp_partial"]:
    r = {}
    for metric, col in (("adopt","commit"),("rmse","rmse"),("sam","sam"),("union","union")):
        v = (df[f"logistic_partial__{col}"] - df[f"{other}__{col}"]).to_numpy(float)
        r[metric] = [float(v.mean()), boot(v)]
    r["union_count"] = int(df.logistic_partial__union.sum() - df[f"{other}__union"].sum())
    R["vs"][other] = r
# residual-map contrasts at fixed scale, from the crossed analysis per-location table is not stored; recompute from crossed json inputs
cr = json.loads((run/"base"/"crossed_scale.json").read_text())
R["crossed_table"] = cr["table"]
# grid-edge validation values
d3 = json.loads((run/"base"/"fold_3_decision.json").read_text())["families"]["gated_ridge_full"]["validation_grid"]
R["fold3_gated_ridge_full_grid"] = [g for g in d3 if g["params"].get("alpha") in (10000.0, 1000000.0)]
# envelope: fold-development DESIS range per band
des = pd.read_csv(sys.argv[3]); bands = [c for c in des.columns if c.startswith("d") and c[1:].isdigit() and c != "d403"]
Y = des[bands].to_numpy(float)
spec = {}
for k in range(5):
    z = np.load(run/"base"/f"spectra_fold_{k}.npz"); idx = z["row_index"]
    for a in ("logistic_global","logistic_full","logistic_partial"):
        spec.setdefault(a, np.full(Y.shape, np.nan))[idx] = z[a]
spec["desis"] = Y
env = {}
for a, x in spec.items():
    out = 0
    for k in range(5):
        dev = sm[f"fold_{k}_role"].to_numpy() == "development"; test = df.fold.to_numpy() == k
        lo, hi = Y[dev].min(0), Y[dev].max(0)
        out += int(((x[test] < lo) | (x[test] > hi)).sum())
    env[a] = out / Y.size * 100
R["envelope_pct"] = env
json.dump(R, open(sys.argv[4], "w"), indent=1); print(json.dumps(R, indent=1)[:3000])
