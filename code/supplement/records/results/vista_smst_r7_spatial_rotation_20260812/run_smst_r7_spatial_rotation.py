#!/usr/bin/env python3
"""Five-fold same-release spatial robustness characterization for SMST."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R4_SCRIPT_SHA256 = "3a14e29f063f5809d941f4b202483bb017345475164c6c10a6eba90cffd770f1"
R4_PREREG_SHA256 = "7618c6c5859cf43e84a7a158121c02872bb519c8d548331012d1c1c6a18ae04a"
CONFIG = {"global_alpha": 1e-2, "decoder_alpha": 1e-1, "residual_alpha": 1e-4}
SEED = "smst-r7-spatial-5fold-20260812:"
HARM_ABSOLUTE, HARM_RELATIVE = 0.25, 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spatial_fold(lon: float, lat: float) -> int:
    block = f"{int(np.floor((lon + 180.0) / .01))}:{int(np.floor((lat + 90.0) / .01))}"
    return int.from_bytes(hashlib.sha256((SEED + block).encode()).digest()[:8], "big") % 5


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4-runner", required=True, type=Path)
    parser.add_argument("--r4-prereg", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    if args.outdir.exists(): raise RuntimeError(f"Output exists: {args.outdir}")
    if sha256(args.r4_runner) != R4_SCRIPT_SHA256 or sha256(args.r4_prereg) != R4_PREREG_SHA256:
        raise RuntimeError("R4 authority source hash mismatch")
    sys.path.insert(0, str(args.r4_runner.parent))
    import run_smst_r4 as core  # noqa: PLC0415
    data = core.load_data(args.data_dir)
    fold = np.asarray([spatial_fold(x, y) for x, y in zip(data.lon, data.lat, strict=True)], dtype=np.int8)
    args.outdir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for held_out in range(5):
        train, test = np.flatnonzero(fold != held_out), np.flatnonzero(fold == held_out)
        support = {str(c): int((data.crop[test] == c).sum()) for c in data.classes}
        if min(support.values()) < 10: raise RuntimeError(f"Fold {held_out} support failure: {support}")
        bundle = core.fit_bundle(data.z_prisma[train], data.y_desis[train], data.crop[train], data.classes, **CONFIG)
        z, y, labels = data.z_prisma[test], data.y_desis[test], data.crop[test]
        global_x, semantic_x, global_idx, margin_global, margin_semantic = core.transports(bundle, z)
        semantic_idx, _ = core.top_and_margin(core.logits(bundle, semantic_x))
        accepted = (semantic_idx == global_idx) & (margin_semantic >= margin_global)
        smst_x = np.where(accepted[:, None], semantic_x, global_x)
        global_pred, semantic_pred = data.classes[global_idx], data.classes[semantic_idx]
        global_metrics = core.metrics(global_pred, labels, global_x, y, data.classes)
        smst_metrics = core.metrics(global_pred, labels, smst_x, y, data.classes)
        global_rmse = global_metrics.pop("per_sample_rmse"); global_sam = global_metrics.pop("per_sample_sam")
        smst_rmse = smst_metrics.pop("per_sample_rmse"); smst_sam = smst_metrics.pop("per_sample_sam")
        harm = ((smst_rmse - global_rmse > HARM_ABSOLUTE) & ((smst_rmse - global_rmse) / np.maximum(global_rmse, 1e-12) > HARM_RELATIVE))
        records.append({"fold": held_out, "train_n": int(train.size), "test_n": int(test.size), "test_class_support": support, "coverage": float(accepted.mean()), "candidate_label_changes": int((semantic_pred != global_pred).sum()), "post_gate_label_changes": 0, "declared_spectral_harms": int(harm.sum()), "delta_macro_f1_vs_global": float(smst_metrics["macro_f1"] - global_metrics["macro_f1"]), "delta_rmse_vs_global": float(smst_metrics["rmse"] - global_metrics["rmse"]), "delta_sam_rad_vs_global": float(smst_metrics["sam_rad"] - global_metrics["sam_rad"]), "global_rmse": float(global_metrics["rmse"]), "smst_rmse": float(smst_metrics["rmse"]), "global_sam_rad": float(global_metrics["sam_rad"]), "smst_sam_rad": float(smst_metrics["sam_rad"]), "global_macro_f1": float(global_metrics["macro_f1"]), "smst_macro_f1": float(smst_metrics["macro_f1"])})
    fields = ["fold", "train_n", "test_n", "coverage", "candidate_label_changes", "post_gate_label_changes", "declared_spectral_harms", "delta_macro_f1_vs_global", "delta_rmse_vs_global", "delta_sam_rad_vs_global", "global_rmse", "smst_rmse", "global_sam_rad", "smst_sam_rad", "global_macro_f1", "smst_macro_f1"]
    with (args.outdir / "R7_FOLD_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    summary = {"schema": "smst-r7-spatial-rotation-v1", "claim_boundary": "Five same-release spatial folds with a frozen R4 configuration; not an independent sensor-pair, regional, seasonal, or population evaluation.", "r4_runner_sha256": sha256(args.r4_runner), "r4_preregistration_sha256": sha256(args.r4_prereg), "seed": SEED, "config": CONFIG, "folds": records, "aggregate": {"folds": 5, "all_post_gate_label_changes_zero": all(x["post_gate_label_changes"] == 0 for x in records), "all_delta_rmse_nonpositive": all(float(x["delta_rmse_vs_global"]) <= 0.0 for x in records), "mean_delta_rmse": float(np.mean([x["delta_rmse_vs_global"] for x in records])), "mean_delta_sam_rad": float(np.mean([x["delta_sam_rad_vs_global"] for x in records])), "mean_coverage": float(np.mean([x["coverage"] for x in records])), "total_declared_spectral_harms": int(sum(x["declared_spectral_harms"] for x in records))}}
    write_json(args.outdir / "R7_SUMMARY.json", summary)
    digests = {p.name: sha256(p) for p in sorted(args.outdir.iterdir()) if p.is_file()}
    (args.outdir / "SHA256SUMS").write_text("\n".join(f"{v}  {k}" for k, v in digests.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
