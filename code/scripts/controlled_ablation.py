#!/usr/bin/env python3
"""Controlled component analysis: one fixed residual map, gate and scale switched one at a time.

For each outer fold, the full-selected and partial-selected logistic residual maps are refit with
their selected parameters (no reselection) and evaluated with and without the label-consistency
gate, at unit scale and at the fold's selected partial scale. The gated outputs must reproduce
crossed_scale.csv exactly.
Usage: python controlled_ablation.py ORIGINAL.py DESIS PRISMA RUN_DIR OUT.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vista_converged  # noqa: E402

o = vista_converged.load(Path(sys.argv[1]))
data = o.load_release(Path(sys.argv[2]), Path(sys.argv[3]))
run = Path(sys.argv[4])
x, y, labels = data["x"], data["y"], data["class_idx"]
held = pd.read_csv(run / "heldout_per_location.csv").set_index("row_index")
cs = pd.read_csv(run / "crossed_scale.csv").set_index("row_index")
rows = []
for part in o.make_partitions(data["lon"], data["lat"]):
    k, dev, test = part["fold"], part["dev"], part["test"]
    dec = json.loads((run / f"fold_{k}_decision.json").read_text())
    s1, fam = dec["stage1"]["params"], dec["families"]
    gs = o.fit_diag_affine(x[dev], y[dev], s1["global_alpha"])
    f = o.fit_logistic(y[dev], labels[dev], s1["decoder_alpha"])
    gd, gt = o.apply_diag(x[dev], gs), o.apply_diag(x[test], gs)
    pdv, pt = f.proba(gd), f.proba(gt)
    gpred = f.predict_idx(gt)
    a_p = fam["logistic_partial"]["params"]["scale"]
    out = {}
    for mname, fam_name in (("full", "logistic_full"), ("partial", "logistic_partial")):
        states = o.fit_weighted_residual(x[dev], y[dev], gd, pdv, fam[fam_name]["params"]["residual_alpha"])
        full = o.apply_weighted_residual(x[test], gt, pt, states)
        for sname, s in (("unit", 1.0), ("partial", a_p)):
            cand = gt + s * (full - gt)
            out[(mname, sname, "nogate")] = o.evaluate(cand, y[test], labels[test], f, gt, gpred)
            g_out, commit, _ = o.gate_output(cand, gt, f)
            out[(mname, sname, "gate")] = o.evaluate(g_out, y[test], labels[test], f, gt, gpred, commit)
            ref = cs.loc[test, f"{mname}_{sname}__d"].to_numpy()
            if not np.allclose(out[(mname, sname, "gate")]["delta_rmse_row"], ref, atol=1e-10):
                raise RuntimeError(f"fold {k}: gated {mname}/{sname} does not reproduce crossed_scale.csv")
    for i, idx in enumerate(test):
        r = {"row_index": int(idx), "fold": k, "outer_cell": held.at[idx, "outer_cell"], "label": int(labels[idx])}
        for (m, s, g), met in out.items():
            key = f"{m}_{s}_{g}"
            r[f"{key}__d"] = float(met["delta_rmse_row"][i]); r[f"{key}__tail"] = int(met["tail"][i])
            r[f"{key}__lab"] = int(met["label_change"][i]); r[f"{key}__c2i"] = int(met["c2i"][i])
            r[f"{key}__union"] = int(met["union"][i]); r[f"{key}__pred"] = int(met["pred"][i])
            r[f"{key}__commit"] = int(met["commit"][i]) if met.get("commit") is not None else 1
        rows.append(r)
df = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
assert len(df) == 16581


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
df.to_csv(Path(sys.argv[5]).with_suffix(".csv"), index=False)
json.dump(res, open(sys.argv[5], "w"), indent=1)
for k, v in res["table"].items():
    print(f"{k:24s} adopt {v['adopt']:6.2f} labels {v['labels']:5d} c2i {v['c2i']:5d} dRMSE {v['delta_rmse']:+.4f} events {v['events']:5d} union {v['union']:5d} F1 {v['f1']:.4f}")
print(json.dumps({k: res[k] for k in ("gate_effect", "scale_effect")}, indent=1))
