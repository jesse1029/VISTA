#!/usr/bin/env python3
"""Fixed component characterization of the already locked R4 SMST gate.

This script deliberately imports the R4 realization and refuses to fit or
select anything.  It is a post-R4 descriptive ablation, not a new primary
experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

R4_SCRIPT_SHA256 = "3a14e29f063f5809d941f4b202483bb017345475164c6c10a6eba90cffd770f1"
R4_PREREG_SHA256 = "7618c6c5859cf43e84a7a158121c02872bb519c8d548331012d1c1c6a18ae04a"
EXPECTED_KEY = "g=1e-02|d=1e-01|r=1e-04"
EXPECTED_MARGIN = 0.0
HARM_ABSOLUTE = 0.25
HARM_RELATIVE = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_row(core, chosen: np.ndarray, predicted: np.ndarray, labels: np.ndarray,
               classes: np.ndarray, global_rmse: np.ndarray, global_sam: np.ndarray,
               accepted: np.ndarray, global_pred: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    values = core.metrics(predicted, labels, chosen, reference, classes)
    rmse = values.pop("per_sample_rmse")
    sam = values.pop("per_sample_sam")
    harm = ((rmse - global_rmse > HARM_ABSOLUTE) &
            ((rmse - global_rmse) / np.maximum(global_rmse, 1e-12) > HARM_RELATIVE))
    correct_to_incorrect = (global_pred == labels) & (predicted != labels)
    return {
        "coverage": float(accepted.mean()),
        "frozen_decoder_label_changes_vs_global": int((predicted != global_pred).sum()),
        "correct_to_incorrect_vs_global": int(correct_to_incorrect.sum()),
        "spectral_harms_vs_global": int(harm.sum()),
        "component_union_harms_vs_global": int((harm | correct_to_incorrect).sum()),
        "delta_rmse_vs_global": float(values["rmse"] - global_rmse.mean()),
        "delta_sam_rad_vs_global": float(values["sam_rad"] - global_sam.mean()),
        "delta_macro_f1_vs_global": None,  # filled after the global row is known
        "accuracy": float(values["accuracy"]),
        "macro_f1": float(values["macro_f1"]),
        "rmse": float(values["rmse"]),
        "sam_rad": float(values["sam_rad"]),
        "_harm": harm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4-runner", required=True, type=Path)
    parser.add_argument("--r4-prereg", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    if args.outdir.exists():
        raise RuntimeError(f"Output exists: {args.outdir}")
    if sha256(args.r4_runner) != R4_SCRIPT_SHA256 or sha256(args.r4_prereg) != R4_PREREG_SHA256:
        raise RuntimeError("R4 authority source hash mismatch")
    sys.path.insert(0, str(args.r4_runner.parent))
    import run_smst_r4 as core  # noqa: PLC0415
    lock = json.loads((args.validation / "FINAL_DECISION_LOCK.json").read_text(encoding="utf-8"))
    selected = lock["selected"]
    if selected["model_key"] != EXPECTED_KEY or float(selected["margin"]) != EXPECTED_MARGIN:
        raise RuntimeError("R4 lock is not the expected frozen configuration")
    if lock["script_sha256"] != R4_SCRIPT_SHA256 or lock["pre_registration_sha256"] != R4_PREREG_SHA256:
        raise RuntimeError("R4 lock authority mismatch")
    data = core.load_data(args.data_dir)
    with (args.development / "development_models.pkl").open("rb") as handle:
        bundle = pickle.load(handle)[EXPECTED_KEY]
    final = core.indices(data, "final")
    z, reference, labels = data.z_prisma[final], data.y_desis[final], data.crop[final]
    global_x, semantic_x, global_idx, margin_global, margin_semantic = core.transports(bundle, z)
    semantic_idx, _ = core.top_and_margin(core.logits(bundle, semantic_x))
    global_pred, semantic_pred = data.classes[global_idx], data.classes[semantic_idx]
    global_metrics = core.metrics(global_pred, labels, global_x, reference, data.classes)
    global_rmse, global_sam = global_metrics.pop("per_sample_rmse"), global_metrics.pop("per_sample_sam")
    predicates = {
        "unconstrained": np.ones(final.size, dtype=bool),
        "label_only": semantic_idx == global_idx,
        "margin_only": margin_semantic >= margin_global,
        "both": (semantic_idx == global_idx) & (margin_semantic >= margin_global),
    }
    rows: dict[str, dict[str, object]] = {}
    for name, accepted in predicates.items():
        chosen = np.where(accepted[:, None], semantic_x, global_x)
        predicted = np.where(accepted, semantic_pred, global_pred)
        rows[name] = metric_row(core, chosen, predicted, labels, data.classes, global_rmse, global_sam, accepted, global_pred, reference)
    base_f1 = float(global_metrics["macro_f1"])
    for value in rows.values():
        value["delta_macro_f1_vs_global"] = float(value["macro_f1"]) - base_f1
        value.pop("_harm")
    args.outdir.mkdir(parents=True)
    with (args.outdir / "R6_PER_LOCATION.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["loc_id", "crop_label", "global_prediction", "semantic_prediction", "global_margin", "semantic_margin", "global_rmse", "semantic_rmse", *[f"accept_{x}" for x in predicates]]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        semantic_rmse = np.sqrt(np.mean((semantic_x - reference) ** 2, axis=1))
        for i, location in enumerate(data.loc_id[final]):
            writer.writerow({"loc_id": int(location), "crop_label": str(labels[i]), "global_prediction": str(global_pred[i]), "semantic_prediction": str(semantic_pred[i]), "global_margin": float(margin_global[i]), "semantic_margin": float(margin_semantic[i]), "global_rmse": float(global_rmse[i]), "semantic_rmse": float(semantic_rmse[i]), **{f"accept_{name}": bool(values[i]) for name, values in predicates.items()}})
    summary = {
        "schema": "smst-r6-fixed-gate-components-v1",
        "claim_boundary": "Post-R4 component characterization on the once-opened R4 final split. It is not a new primary test, independent replication, or safety certificate.",
        "r4_runner_sha256": sha256(args.r4_runner),
        "r4_preregistration_sha256": sha256(args.r4_prereg),
        "r4_decision_lock_sha256": sha256(args.validation / "FINAL_DECISION_LOCK.json"),
        "selected_model_key": EXPECTED_KEY,
        "selected_margin": EXPECTED_MARGIN,
        "final_rows": int(final.size),
        "global": global_metrics,
        "variants": rows,
    }
    dump(args.outdir / "R6_SUMMARY.json", summary)
    digests = {p.name: sha256(p) for p in sorted(args.outdir.iterdir()) if p.is_file()}
    (args.outdir / "SHA256SUMS").write_text("\n".join(f"{digest}  {name}" for name, digest in digests.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
