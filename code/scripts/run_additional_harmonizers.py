#!/usr/bin/env python3
"""Additional cross-sensor harmonizers under the VISTA protocol.

Adds partial least squares (PLS), kernel ridge regression with an RBF kernel (Nystroem
approximation with a fitted ridge intercept), and a
multilayer perceptron (MLP) to the buffered five-fold comparison.  Every harmonizer maps the
59-band PRISMA input to the DESIS reference and is evaluated exactly like ridge regression and
PDS in run_common_utility_expanded.py:

* the same signed-hash outer folds, 0.02-degree buffer, and development/validation split;
* the fold's selected global fallback and logistic classifier (stage 1 is reselected with the
  published rule, which reproduces the published stage-1 decisions);
* an ungated configuration, and gated full (scale 1) and partial (scale 0.10-0.75)
  configurations, each selected separately on validation outputs with the published key
  (higher macro-F1, lower RMSE, fewer union events, lower SAM, parameter string);
* held-out rows scored only after every fold decision is fixed.

Usage:
  python run_additional_harmonizers.py --original path/to/original_common_utility.py \
      --desis desis_2021.csv --prisma prisma_2021.csv --out OUT_DIR [--families pls,krr,mlp]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SCALES_PARTIAL = [0.10, 0.25, 0.50, 0.75]
O = None  # original module, set in main()


sys.path.insert(0, str(Path(__file__).resolve().parent))
import vista_converged  # noqa: E402


def load_original(path: Path, converged: bool):
    return vista_converged.load(path, converged=converged)


class Standardized:
    """Standardize inputs and targets on development rows, then fit a 59-to-59 regressor."""

    def __init__(self, make):
        self.make = make

    def fit(self, x, y):
        self.sx = StandardScaler().fit(x)
        self.sy = StandardScaler().fit(y)
        self.model = self.make()
        self.model.fit(self.sx.transform(x), self.sy.transform(y))
        return self

    def predict(self, x):
        return self.sy.inverse_transform(self.model.predict(self.sx.transform(x)))


class NystroemRidge:
    def __init__(self, gamma, alpha, components, seed):
        self.feat = Nystroem(kernel="rbf", gamma=gamma, n_components=components, random_state=seed)
        self.ridge = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, x, y):
        self.ridge.fit(self.feat.fit_transform(x), y)
        return self

    def predict(self, x):
        return self.ridge.predict(self.feat.transform(x))


def family_grid(name: str):
    """(params, factory) pairs; each factory returns an unfitted 59-to-59 regressor."""
    if name == "pls":
        return [({"n_components": k},
                 lambda k=k: Standardized(lambda: PLSRegression(n_components=k, scale=False, max_iter=1000)))
                for k in (5, 10, 20, 30, 40, 50, 59)]
    if name == "krr":
        out = []
        for g in (0.1, 1.0, 10.0):
            for a in (1e-5, 1e-3, 1e-1, 10.0):
                out.append(({"gamma_x_bands": g, "alpha": a},
                            lambda g=g, a=a: Standardized(lambda: NystroemRidge(g / 59.0, a, 2000, 20261016))))
        return out
    if name == "mlp":
        return [({"hidden": [256, 256], "alpha": a},
                 lambda a=a: Standardized(lambda: MLPRegressor(
                     hidden_layer_sizes=(256, 256), activation="relu", solver="adam", alpha=a,
                     batch_size=256, learning_rate_init=1e-3, max_iter=400, early_stopping=True,
                     validation_fraction=0.1, n_iter_no_change=20, random_state=20261016)))
                for a in (1e-4, 1e-2)]
    if name == "pds_rule":
        # Ungated PDS on the published expanded window grid, selected by the published rule.
        return [({"ridge_alpha": 1e-2, "window": w}, lambda w=w: PDS(w)) for w in (1, 3, 7, 11, 15, 21, 31, 41)]
    raise ValueError(name)


class PDS:
    def __init__(self, window):
        self.window = window

    def fit(self, x, y):
        self.states = O.fit_pds(x, y, self.window, 1e-2)
        return self

    def predict(self, x):
        return O.apply_pds(x, self.states)


def run_fold(o, data, part, families, lock_stage1):
    fold = part["fold"]
    dev, val, test = part["dev"], part["val"], part["test"]
    x, y, labels = data["x"], data["y"], data["class_idx"]

    # Stage 1: identical candidate set and key as the published script.
    cands = []
    for ga in [1e-6, 1e-4, 1e-2, 1.0]:
        gs = o.fit_diag_affine(x[dev], y[dev], ga)
        gv = o.apply_diag(x[val], gs)
        for da in [1e-5, 1e-3, 1e-1]:
            dec = o.fit_logistic(y[dev], labels[dev], da)
            m = o.evaluate(gv, y[val], labels[val], dec, gv)
            p = {"global_alpha": ga, "decoder_alpha": da}
            cands.append((o.selection_key(m, p), p, gs, dec))
    cands.sort(key=lambda z: z[0])
    chosen = cands[0]
    if lock_stage1 is not None:
        # Same as --lock-published in the published script: reuse the published stage-1 state.
        chosen = next(z for z in cands if z[1] == lock_stage1)
    _, stage1, gstate, logistic = chosen
    rule_stage1 = cands[0][1]

    g_val = o.apply_diag(x[val], gstate)
    g_test = o.apply_diag(x[test], gstate)
    gpred_val = logistic.predict_idx(g_val)
    gpred_test = logistic.predict_idx(g_test)

    decisions, outputs, spectra = {}, {}, {}
    for fam in families:
        t0 = time.time()
        fitted = []
        for params, factory in family_grid(fam):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = factory().fit(x[dev], y[dev])
            fitted.append((params, model, model.predict(x[val])))

        # ungated
        lst = []
        for params, model, raw_val in fitted:
            m = o.evaluate(raw_val, y[val], labels[val], logistic, g_val, gpred_val)
            lst.append((o.selection_key(m, params), params, model, m))
        lst.sort(key=lambda z: z[0])
        _, params, model, m = lst[0]
        decisions[fam] = {"params": params, "validation": o.stripped_metrics(m),
                          "validation_grid": [{"params": p, **o.stripped_metrics(mm)} for _, p, _, mm in lst]}
        outputs[fam] = o.evaluate(model.predict(x[test]), y[test], labels[test], logistic, g_test, gpred_test)

        if fam == "pds_rule":
            print(f"[fold {fold}] pds_rule done: {decisions[fam]['params']}", flush=True)
            continue
        # gated full and partial
        for variant, scales in ((f"gated_{fam}_full", [1.0]), (f"gated_{fam}_partial", SCALES_PARTIAL)):
            lst = []
            for params, model, raw_val in fitted:
                for s in scales:
                    cand = g_val + float(s) * (raw_val - g_val)
                    out, commit, _ = o.gate_output(cand, g_val, logistic)
                    m = o.evaluate(out, y[val], labels[val], logistic, g_val, gpred_val, commit)
                    p = {**params, "scale": float(s)}
                    lst.append((o.selection_key(m, p), p, model, m))
            lst.sort(key=lambda z: z[0])
            _, p, model, m = lst[0]
            decisions[variant] = {"params": p, "validation": o.stripped_metrics(m),
                                  "validation_grid": [{"params": pp, **o.stripped_metrics(mm)} for _, pp, _, mm in lst]}
            raw_test = model.predict(x[test])
            cand = g_test + float(p["scale"]) * (raw_test - g_test)
            out, commit, _ = o.gate_output(cand, g_test, logistic)
            outputs[variant] = o.evaluate(out, y[test], labels[test], logistic, g_test, gpred_test, commit)
            if variant.endswith("_partial"):
                spectra[variant] = out
        print(f"[fold {fold}] {fam} done in {time.time() - t0:.0f}s: "
              + ", ".join(f"{k}={v['params']}" for k, v in decisions.items() if fam in k), flush=True)

    rows = []
    for pos, idx in enumerate(test):
        row = {"row_index": int(idx), "loc_id": int(data["loc_id"][idx]), "fold": int(fold)}
        for name, met in outputs.items():
            for key, col in (("pred", "pred"), ("rmse_row", "rmse"), ("sam_row", "sam"),
                             ("delta_rmse_row", "delta_rmse"), ("delta_sam_row", "delta_sam"),
                             ("tail", "tail"), ("c2i", "c2i"), ("union", "union"),
                             ("label_change", "label_change"), ("commit", "commit")):
                v = met[key][pos]
                row[f"{name}__{col}"] = float(v) if col in ("rmse", "sam", "delta_rmse", "delta_sam") else int(v)
        rows.append(row)
    return {"fold": fold, "stage1": stage1, "stage1_rule_argmin_here": rule_stage1, "families": decisions}, rows, spectra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--desis", type=Path, required=True)
    ap.add_argument("--prisma", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--families", default="pds_rule,pls,krr,mlp")
    ap.add_argument("--published-decisions", type=Path, default=None,
                    help="directory with fold_k_decision.json whose stage-1 state is reused")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="do not merge fold files; used by parallel workers before an explicit merge")
    ap.add_argument("--archived-solver", action="store_true",
                    help="keep the archived L-BFGS classifier instead of the converged solver")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global O
    o = O = load_original(args.original, converged=not args.archived_solver)
    data = o.load_release(args.desis, args.prisma)
    parts = o.make_partitions(data["lon"], data["lat"])
    fams = [f for f in args.families.split(",") if f]
    records, frames = [], []
    for part in [parts[int(v)] for v in args.folds.split(",")]:
        lock = None
        if args.published_decisions:
            d = json.loads((args.published_decisions / f"fold_{part['fold']}_decision.json").read_text())
            lock = d["stage1"]["params"]
        rec, rows, spectra = run_fold(o, data, part, fams, lock)
        (args.out / f"fold_{part['fold']}_decision.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
        pd.DataFrame(rows).to_csv(args.out / f"fold_{part['fold']}_additional.csv", index=False)
        np.savez_compressed(args.out / f"spectra_additional_fold_{part['fold']}.npz", row_index=part["test"], **spectra)
    files = sorted(args.out.glob("fold_*_additional.csv"))
    if not args.no_aggregate and len(files) == 5:
        df = pd.concat([pd.read_csv(f) for f in files]).sort_values("row_index").reset_index(drop=True)
        assert len(df) == 16581 and df.row_index.nunique() == 16581
        df.to_csv(args.out / "heldout_additional.csv", index=False)
        print("written", args.out / "heldout_additional.csv")


if __name__ == "__main__":
    main()
