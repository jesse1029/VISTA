#!/usr/bin/env python3
"""Fixed-map correction-scale analysis (Table II and Supplement S5) for a rerun.

For each outer fold, the full-selected and partial-selected residual maps of logistic VISTA are
refit with their selected parameters (no reselection) and evaluated at unit scale and at the
fold's selected partial scale.  The two diagonal combinations must reproduce the rerun's
logistic_full and logistic_partial outputs exactly.

Usage: python crossed_scale_analysis.py ORIGINAL.py DESIS PRISMA RUN_DIR OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vista_converged  # noqa: E402

o = vista_converged.load(Path(sys.argv[1]))
data = o.load_release(Path(sys.argv[2]), Path(sys.argv[3]))
run = Path(sys.argv[4])
parts = o.make_partitions(data["lon"], data["lat"])
x, y, labels = data["x"], data["y"], data["class_idx"]
held = pd.read_csv(run / "heldout_per_location.csv").set_index("row_index")

rows = []
for part in parts:
    k, dev, test = part["fold"], part["dev"], part["test"]
    dec = json.loads((run / f"fold_{k}_decision.json").read_text())
    s1, fam = dec["stage1"]["params"], dec["families"]
    gstate = o.fit_diag_affine(x[dev], y[dev], s1["global_alpha"])
    logistic = o.fit_logistic(y[dev], labels[dev], s1["decoder_alpha"])
    g_dev, g_test = o.apply_diag(x[dev], gstate), o.apply_diag(x[test], gstate)
    p_dev, p_test = logistic.proba(g_dev), logistic.proba(g_test)
    gpred = logistic.predict_idx(g_test)
    a_p = fam["logistic_partial"]["params"]["scale"]
    maps = {"full": fam["logistic_full"]["params"]["residual_alpha"],
            "partial": fam["logistic_partial"]["params"]["residual_alpha"]}
    out = {}
    for mname, ra in maps.items():
        states = o.fit_weighted_residual(x[dev], y[dev], g_dev, p_dev, ra)
        full = o.apply_weighted_residual(x[test], g_test, p_test, states)
        for sname, s in (("unit", 1.0), ("partial", a_p)):
            cand = g_test + s * (full - g_test)
            o_, commit, _ = o.gate_output(cand, g_test, logistic)
            out[(mname, sname)] = o.evaluate(o_, y[test], labels[test], logistic, g_test, gpred, commit)
    for (mname, sname), ref in ((("full", "unit"), "logistic_full"), (("partial", "partial"), "logistic_partial")):
        r = held.loc[test, f"{ref}__rmse"].to_numpy()
        if not np.allclose(out[(mname, sname)]["rmse_row"], r, atol=1e-10):
            raise RuntimeError(f"fold {k}: {mname}/{sname} does not reproduce {ref}")
    for i, idx in enumerate(test):
        row = {"row_index": int(idx), "fold": k, "outer_cell": held.at[idx, "outer_cell"]}
        for (mname, sname), m in out.items():
            key = f"{mname}_{sname}"
            row[f"{key}__d"] = float(m["delta_rmse_row"][i])
            row[f"{key}__tail"] = int(m["tail"][i])
            row[f"{key}__commit"] = int(m["commit"][i])
        rows.append(row)
df = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def boot(values, seed=20260821, reps=2000, denom=None):
    cells = df.outer_cell.to_numpy()
    uniq = np.unique(cells)
    idx = {c: np.where(cells == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if denom is None:
            vals.append(values[pick].mean())
        else:
            w = denom[pick]
            vals.append(values[pick][w].mean() if w.any() else np.nan)
    return [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])]


res = {"table": {}, "partition": {}, "contrasts": {}}
for key in ("full_unit", "full_partial", "partial_unit", "partial_partial"):
    res["table"][key] = {"adopt_pct": float(df[f"{key}__commit"].mean() * 100),
                         "delta_rmse": float(df[f"{key}__d"].mean()),
                         "events": int(df[f"{key}__tail"].sum())}
for mname in ("full", "partial"):
    u, p = f"{mname}_unit", f"{mname}_partial"
    cu, cp = df[f"{u}__commit"].to_numpy(bool), df[f"{p}__commit"].to_numpy(bool)
    nested = bool(np.all(cp[cu]))
    common, new = cu, cp & ~cu
    dd = df[f"{p}__d"].to_numpy() - df[f"{u}__d"].to_numpy()
    res["partition"][mname] = {
        "nested": nested, "common": int(common.sum()), "new": int(new.sum()), "never": int((~cp).sum()),
        "common_mean_change": float(dd[common].mean()), "common_ci": boot(dd, denom=common),
        "new_mean_change": float(dd[new].mean()), "new_ci": boot(dd, denom=new),
        "new_improved": int((dd[new] < 0).sum()),
        "events_common_unit": int(df[f"{u}__tail"].to_numpy()[common].sum()),
        "events_common_partial": int(df[f"{p}__tail"].to_numpy()[common].sum()),
        "events_new_partial": int(df[f"{p}__tail"].to_numpy()[new].sum()),
    }
    res["contrasts"][mname] = {
        "adopt": [float((cp.astype(float) - cu).mean()), boot(cp.astype(float) - cu)],
        "rmse": [float(dd.mean()), boot(dd)],
        "event_rate": [float((df[f"{p}__tail"] - df[f"{u}__tail"]).mean()),
                       boot((df[f"{p}__tail"] - df[f"{u}__tail"]).to_numpy(float))],
    }
res["map_contrasts"] = {}
for sname in ("unit", "partial"):
    a, b = f"partial_{sname}", f"full_{sname}"
    res["map_contrasts"][sname] = {
        "adopt": [float((df[f"{a}__commit"] - df[f"{b}__commit"]).mean()), boot((df[f"{a}__commit"] - df[f"{b}__commit"]).to_numpy(float))],
        "rmse": [float((df[f"{a}__d"] - df[f"{b}__d"]).mean()), boot((df[f"{a}__d"] - df[f"{b}__d"]).to_numpy(float))],
        "event_rate": [float((df[f"{a}__tail"] - df[f"{b}__tail"]).mean()), boot((df[f"{a}__tail"] - df[f"{b}__tail"]).to_numpy(float))],
    }
df.to_csv(Path(sys.argv[5]).with_suffix(".csv"), index=False)
json.dump(res, open(sys.argv[5], "w"), indent=1)
print(json.dumps(res, indent=1))
