#!/usr/bin/env python3
"""Rule-based rerun of the archived buffered five-fold comparison with a converged classifier.

Runs the archived run_fold() for every outer fold with the expanded grids and no published
locks, so every configuration is selected by the published validation rule.  Held-out output
spectra of the configurations used in the figures are captured and checked against the
per-location RMSE written by the archived code.

Usage:
  python run_vista_comparison.py --original original_common_utility.py \
      --desis desis_2021.csv --prisma prisma_2021.csv --out OUT_DIR [--folds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vista_converged  # noqa: E402

NAMES = ["logistic_global", "rf_global", "full_ridge", "pds", "soft_residual",
         "gated_ridge_full", "gated_ridge_partial", "gated_pds_full", "gated_pds_partial",
         "logistic_full", "logistic_partial", "rf_full", "rf_partial"]
KEEP = ["logistic_global", "logistic_full", "logistic_partial", "gated_ridge_partial",
        "gated_pds_partial", "rf_partial", "full_ridge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--desis", type=Path, required=True)
    ap.add_argument("--prisma", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    o = vista_converged.load(args.original)
    data = o.load_release(args.desis, args.prisma)
    parts = o.make_partitions(data["lon"], data["lat"])
    manifest = pd.DataFrame({"row_index": np.arange(len(data["x"])), "loc_id": data["loc_id"]})
    for part in parts:
        role = np.full(len(data["x"]), "excluded", dtype=object)
        for source, label in (("dev", "development"), ("val", "validation"), ("test", "test")):
            role[part[source]] = label
        manifest[f"fold_{part['fold']}_role"] = role
    temporary_manifest = args.out / f"split_manifest.{os.getpid()}.tmp"
    manifest.to_csv(temporary_manifest, index=False)
    temporary_manifest.replace(args.out / "split_manifest.csv")

    for k in [int(v) for v in args.folds.split(",")]:
        part = parts[k]
        test = part["test"]
        y_test = data["y"][test]
        captured = []
        evaluate = o.evaluate

        def capture(out, ref, *a, **kw):
            if out.shape == y_test.shape and np.array_equal(ref, y_test):
                captured.append(np.array(out, copy=True))
            return evaluate(out, ref, *a, **kw)

        o.evaluate = capture
        try:
            print(f"[fold {k}] start", flush=True)
            record = o.run_fold(data, part, expanded=True, lock_published=False)
        finally:
            o.evaluate = evaluate
        if len(captured) != len(NAMES):
            raise RuntimeError(f"fold {k}: captured {len(captured)} held-out outputs, expected {len(NAMES)}")
        rows = pd.DataFrame(record.pop("rows"))
        spectra = {}
        for name, out in zip(NAMES, captured):
            e = np.sqrt(((out - y_test) ** 2).mean(1))
            if not np.allclose(e, rows[f"{name}__rmse"].to_numpy(), atol=1e-12):
                raise RuntimeError(f"fold {k}: captured spectra do not match {name}")
            if name in KEEP:
                spectra[name] = out
        np.savez_compressed(args.out / f"spectra_fold_{k}.npz", row_index=test, **spectra)
        rows.to_csv(args.out / f"fold_{k}_heldout.csv", index=False)
        (args.out / f"fold_{k}_decision.json").write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"[fold {k}] done: stage1 {record['stage1']['params']}", flush=True)

    files = sorted(args.out.glob("fold_*_heldout.csv"))
    if len(files) == 5:
        heldout = pd.concat([pd.read_csv(f) for f in files]).sort_values("row_index").reset_index(drop=True)
        assert len(heldout) == 16581 and heldout.row_index.nunique() == 16581
        heldout.to_csv(args.out / "heldout_per_location.csv", index=False)
        summary = o.summarize(heldout)
        summary["classifier_solver"] = vista_converged.SOLVER
        summary["runtime"] = {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
        summary["inputs"] = {"desis_sha256": o.sha256(args.desis), "prisma_sha256": o.sha256(args.prisma)}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("written", args.out / "summary.json")


if __name__ == "__main__":
    main()
