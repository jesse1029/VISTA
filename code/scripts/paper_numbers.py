#!/usr/bin/env python3
"""Every quantity the manuscript and supplement report from the rule-based rerun.

Usage: python paper_numbers.py RUN_DIR DESIS_CSV OUT.json
RUN_DIR contains base/ (run_vista_comparison.py) and ext/ (run_additional_harmonizers.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

run, desis_csv, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
base = pd.read_csv(run / "base" / "heldout_per_location.csv").sort_values("row_index").reset_index(drop=True)
ext = pd.read_csv(run / "ext" / "heldout_additional.csv").sort_values("row_index").reset_index(drop=True)
assert (base.row_index.values == ext.row_index.values).all()
df = pd.concat([base, ext.drop(columns=["row_index", "loc_id", "fold"])], axis=1)
N = len(df)
labels = df.class_idx.to_numpy()

des = pd.read_csv(desis_csv)
bands = [c for c in des.columns if c.startswith("d") and c[1:].isdigit() and c != "d403"]
wl = np.array([int(c[1:]) for c in bands])
Y = des[bands].to_numpy(float)

# ---------------------------------------------------------------- spectra
spec = {}
for k in range(5):
    for f in (run / "base" / f"spectra_fold_{k}.npz", run / "ext" / f"spectra_additional_fold_{k}.npz"):
        z = np.load(f)
        idx = z["row_index"]
        for a in z.files:
            if a == "row_index":
                continue
            spec.setdefault(a, np.full((N, len(wl)), np.nan))[idx] = z[a]
for a, arr in spec.items():
    assert not np.isnan(arr).any(), a
    e = np.sqrt(((arr - Y) ** 2).mean(1))
    assert np.allclose(e, df[f"{a}__rmse"].to_numpy(), atol=1e-9), a


def boot_mean(values, seed=20260816, reps=2000, mask=None):
    cells = df.outer_cell.to_numpy()
    uniq = np.unique(cells)
    pos = {c: np.where(cells == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = np.concatenate([pos[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if mask is None:
            vals.append(values[pick].mean())
        else:
            m = mask[pick]
            vals.append(values[pick][m].mean() if m.any() else np.nan)
    return [float(v) for v in np.nanpercentile(vals, [2.5, 97.5])]


R = {}
UNGATED = ["full_ridge", "pds", "pls", "krr", "mlp", "soft_residual"]
GATED = ["gated_ridge_partial", "gated_pds_partial", "gated_pls_partial", "gated_krr_partial",
         "gated_mlp_partial", "logistic_partial", "rf_partial"]
FULLS = ["gated_ridge_full", "gated_pds_full", "gated_pls_full", "gated_krr_full", "gated_mlp_full",
         "logistic_full", "rf_full"]
for m in UNGATED + GATED + FULLS + ["logistic_global", "rf_global"]:
    d = df[f"{m}__delta_rmse"].to_numpy()
    k1 = int(np.ceil(0.01 * N))
    r = {
        "adopt": float(df[f"{m}__commit"].mean() * 100),
        "fallback_pct": float((d == 0).mean() * 100),
        "f1": float(f1_score(labels, df[f"{m}__pred"], labels=np.arange(5), average="macro", zero_division=0)),
        "mean": float(d.mean()), "mean_ci": boot_mean(d),
        "dsam": float(df[f"{m}__delta_sam"].mean()), "dsam_ci": boot_mean(df[f"{m}__delta_sam"].to_numpy(), seed=20260817),
        "median": float(np.median(d)), "improved": float((d < 0).mean() * 100),
        "worst1": float(np.sort(d)[-k1:].mean()), "max": float(d.max()), "p90": float(np.quantile(d, 0.9)),
        "over05": float((d > 0.5).mean() * 100),
        "labels": int(df[f"{m}__label_change"].sum()), "c2i": int(df[f"{m}__c2i"].sum()),
        "events": int(df[f"{m}__tail"].sum()), "union": int(df[f"{m}__union"].sum()),
        "rmse_abs": float(df[f"{m}__rmse"].mean()), "sam_abs": float(df[f"{m}__sam"].mean()),
    }
    if m in spec:
        band = np.sqrt(((spec[m] - Y) ** 2).mean(0)) - np.sqrt(((spec["logistic_global"] - Y) ** 2).mean(0))
        r["bands_improved"] = int((band < 0).sum())
        r["band_below500_mean"] = float(band[wl < 500].mean())
        r["n_below500"] = int((wl < 500).sum())
        keep = wl >= 500
        e_keep = np.sqrt(((spec[m][:, keep] - Y[:, keep]) ** 2).mean(1))
        g_keep = np.sqrt(((spec["logistic_global"][:, keep] - Y[:, keep]) ** 2).mean(1))
        r["mean_without_below500"] = float((e_keep - g_keep).mean())
    R[m] = r

# ---------------------------------------------------------------- per crop and per fold
crop = {}
for c, g in df.groupby("crop"):
    crop[c] = {"n": int(len(g)), "cells": int(g.outer_cell.nunique())}
    for m in GATED:
        d = g[f"{m}__delta_rmse"].to_numpy()
        crop[c][m] = {"mean": float(d.mean()), "improved": float((d < 0).mean() * 100),
                      "event_rate": float(g[f"{m}__tail"].mean() * 100), "adopt": float(g[f"{m}__commit"].mean() * 100),
                      "fallback": float((d == 0).mean() * 100)}
    sub = df.crop.to_numpy() == c
    crop[c]["vista_ci"] = boot_mean(df["logistic_partial__delta_rmse"].to_numpy(), mask=sub)
    crop[c]["fallback_rmse"] = float(g.logistic_global__rmse.mean())
    crop[c]["vista_rmse"] = float(g.logistic_partial__rmse.mean())
R["crop"] = crop
# balanced five-crop mean and interval
cells = df.outer_cell.to_numpy(); uniq = np.unique(cells); pos = {c: np.where(cells == c)[0] for c in uniq}
rng = np.random.default_rng(20260816); bal = []
crops = sorted(crop)
dv = df.logistic_partial__delta_rmse.to_numpy(); cr = df.crop.to_numpy()
for _ in range(2000):
    pick = np.concatenate([pos[c] for c in rng.choice(uniq, len(uniq), replace=True)])
    bal.append(np.mean([dv[pick][cr[pick] == c].mean() for c in crops]))
R["balanced"] = {"fallback": float(np.mean([crop[c]["fallback_rmse"] for c in crops])),
                 "vista": float(np.mean([crop[c]["vista_rmse"] for c in crops])),
                 "delta": float(np.mean([crop[c]["logistic_partial"]["mean"] for c in crops])),
                 "ci": [float(v) for v in np.percentile(bal, [2.5, 97.5])]}
fold = {}
for k, g in df.groupby("fold"):
    fold[int(k)] = {m: {"mean": float(g[f"{m}__delta_rmse"].mean()), "adopt": float(g[f"{m}__commit"].mean() * 100),
                        "events": int(g[f"{m}__tail"].sum()), "event_rate": float(g[f"{m}__tail"].mean() * 100),
                        "improved": float((g[f"{m}__delta_rmse"] < 0).mean() * 100)} for m in GATED}
    fold[int(k)]["n"] = int(len(g))
R["fold"] = fold

# ---------------------------------------------------------------- range and smoothness (S6)
splits = None
phys = {}
for name in ("logistic_global", "logistic_full", "logistic_partial"):
    phys[name] = spec[name]
phys["desis"] = Y
phys_out = {}
for name, x in phys.items():
    s = np.diff(x, axis=1) / np.diff(wl)
    c = np.diff(s, axis=1) / ((wl[2:] - wl[:-2]) / 2.0)
    phys_out[name] = {"below0": float((x < 0).mean() * 100), "above100": float((x > 100).mean() * 100),
                      "slope": float(np.abs(s).mean()), "curv": float(np.abs(c).mean())}
R["physical"] = phys_out

# ---------------------------------------------------------------- qualitative cases (Fig. 5)
dvv, drr = df.logistic_partial__delta_rmse, df.gated_ridge_partial__delta_rmse
ad = df[df.logistic_partial__commit == 1]
med = ad.loc[(ad.logistic_partial__delta_rmse - ad.logistic_partial__delta_rmse.median()).abs().idxmin()]
cases = []
for role, row in (("largest VISTA improvement", df.loc[dvv.idxmin()]), ("median adopted VISTA", med),
                  ("largest VISTA worsening", df.loc[dvv.idxmax()]), ("largest ridge worsening", df.loc[drr.idxmax()])):
    cases.append({"role": role, "loc_id": int(row.loc_id), "row_index": int(row.row_index), "fold": int(row.fold),
                  "crop": row.crop, "G": float(row.logistic_global__rmse), "V": float(row.logistic_partial__rmse),
                  "R": float(row.gated_ridge_partial__rmse), "V_adopt": int(row.logistic_partial__commit),
                  "R_adopt": int(row.gated_ridge_partial__commit)})
R["cases"] = cases
r_case = df.loc[drr.idxmax()]
ri = int(r_case.row_index)
nir = wl >= 750
R["ridge_case_nir_overshoot"] = float((spec["gated_ridge_partial"][ri] - Y[ri])[nir].mean())
R["vista_case_nir_offset"] = float(np.abs(spec["logistic_partial"][ri] - Y[ri])[nir].mean())
R["neg_bands_vista_pct"] = phys_out["logistic_partial"]["below0"]
R["neg_bands_desis_pct"] = phys_out["desis"]["below0"]

# ---------------------------------------------------------------- decisions
dec = {}
for k in range(5):
    b = json.loads((run / "base" / f"fold_{k}_decision.json").read_text())
    e = json.loads((run / "ext" / f"fold_{k}_decision.json").read_text())
    assert b["stage1"]["params"] == e["stage1"]
    dec[k] = {"stage1": b["stage1"]["params"],
              **{f: v["params"] for f, v in b["families"].items()},
              **{f: v["params"] for f, v in e["families"].items()}}
R["decisions"] = dec
R["paired"] = json.loads((run / "base" / "summary.json").read_text())["paired"]
R["crossed"] = json.loads((run / "base" / "crossed_scale.json").read_text())
R["wl_below500"] = [int(v) for v in wl[wl < 500]]
json.dump(R, open(out_path, "w"), indent=1)

# ---------------------------------------------------------------- print the scorecard
print(f"{'method':22s} {'lab':>5s} {'c2i':>5s} {'adopt':>6s} {'impr':>5s} {'median':>7s} {'mean':>7s} {'CI':>17s} {'events':>6s} {'max':>5s} {'w1%':>5s} {'bands':>5s} {'<500':>7s} {'no<500':>7s}")
for m in UNGATED + GATED:
    r = R[m]
    print(f"{m:22s} {r['labels']:5d} {r['c2i']:5d} {r['adopt']:6.2f} {r['improved']:5.1f} {r['median']:+7.4f} {r['mean']:+7.4f} "
          f"[{r['mean_ci'][0]:+.3f},{r['mean_ci'][1]:+.3f}] {r['events']:6d} {r['max']:5.2f} {r['worst1']:5.2f} "
          f"{r.get('bands_improved', -1):5d} {r.get('band_below500_mean', np.nan):+7.3f} {r.get('mean_without_below500', np.nan):+7.4f}")
