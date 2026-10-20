#!/usr/bin/env python3
"""R5 geographically separated, fixed-configuration SMST stress test.

This runner deliberately imports the immutable R4 implementation so that the
only new degree of freedom is the pre-registered west/east split.  It does not
perform model or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLD_LONGITUDE = -120.2425085
GAP_DEGREES = 0.01
CONFIG = {"global_alpha": 1e-2, "decoder_alpha": 1e-1, "residual_alpha": 1e-4}
MARGIN = 0.0
EXPECTED_INPUTS = {
    "desis_2021.csv": "85ac21e38fb990a75714f90be0d2ced643275adf4e46570f326e765709c48ccc",
    "prisma_2021.csv": "0c6ef3dff0e5eff1c3e9ebb349494465d4459ee189c2aeb089af4e7622822647",
    "desis_prisma_2021_paper_reconciled.xml": "1453cf193438e59ad61d4cdde2eb4c089c1c28c843517585f7eb47c0f4e5aa46",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_r4(path: Path):
    spec = importlib.util.spec_from_file_location("smst_r4_locked", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import the locked R4 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_direction(r4: Any, data: Any, train_mask: np.ndarray, test_mask: np.ndarray, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(f"Empty regional arm: {name}")
    for side, mask in (("train", train_mask), ("test", test_mask)):
        counts = {str(c): int((data.crop[mask] == c).sum()) for c in data.classes}
        if min(counts.values()) < 20:
            raise RuntimeError(f"{name} {side} lacks class support: {counts}")
    bundle = r4.fit_bundle(
        data.z_prisma[train_mask], data.y_desis[train_mask], data.crop[train_mask], data.classes,
        CONFIG["global_alpha"], CONFIG["decoder_alpha"], CONFIG["residual_alpha"],
    )
    z, y, label = data.z_prisma[test_mask], data.y_desis[test_mask], data.crop[test_mask]
    global_x, semantic_x, global_idx, margin_global, margin_semantic = r4.transports(bundle, z)
    semantic_idx, _ = r4.top_and_margin(r4.logits(bundle, semantic_x))
    accepted = (semantic_idx == global_idx) & (margin_semantic >= margin_global + MARGIN)
    smst_x = np.where(accepted[:, None], semantic_x, global_x)
    pred_global = data.classes[global_idx]
    pred_semantic = data.classes[semantic_idx]
    identity_idx, _ = r4.top_and_margin(r4.logits(bundle, z))
    pred_identity = data.classes[identity_idx]
    label_idx = r4.class_index(label, data.classes)
    residual = z * bundle["residual_a"][label_idx] + bundle["residual_b"][label_idx]
    oracle_x = global_x + residual
    oracle_idx, _ = r4.top_and_margin(r4.logits(bundle, oracle_x))
    scores = {
        "identity": r4.metrics(pred_identity, label, z, y, data.classes),
        "global_ridge": r4.metrics(pred_global, label, global_x, y, data.classes),
        "unconstrained_semantic": r4.metrics(pred_semantic, label, semantic_x, y, data.classes),
        "smst": r4.metrics(pred_global, label, smst_x, y, data.classes),
        "true_label_oracle_diagnostic": r4.metrics(data.classes[oracle_idx], label, oracle_x, y, data.classes),
    }
    global_rmse = scores["global_ridge"].pop("per_sample_rmse")
    for values in scores.values():
        values.pop("per_sample_rmse", None)
        values.pop("per_sample_sam", None)
    smst_rmse = np.sqrt(np.mean((smst_x - y) ** 2, axis=1))
    spectral_harm = (smst_rmse - global_rmse > 0.25) & ((smst_rmse - global_rmse) / np.maximum(global_rmse, 1e-12) > 0.05)
    summary = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_class_counts": {str(c): int((data.crop[train_mask] == c).sum()) for c in data.classes},
        "test_class_counts": {str(c): int((data.crop[test_mask] == c).sum()) for c in data.classes},
        "metrics": scores,
        "paired_vs_global": {
            "coverage": float(accepted.mean()),
            "semantic_label_changes_before_gate": int((pred_semantic != pred_global).sum()),
            "semantic_label_changes_after_gate": 0,
            "correct_to_incorrect_vs_global": 0,
            "spectral_harms_vs_global": int(spectral_harm.sum()),
            "smst_minus_global_rmse": float(scores["smst"]["rmse"] - scores["global_ridge"]["rmse"]),
            "smst_minus_global_sam_rad": float(scores["smst"]["sam_rad"] - scores["global_ridge"]["sam_rad"]),
            "smst_minus_global_macro_f1": 0.0,
        },
    }
    rows = []
    for loc, crop, gp, sp, acc, gr, sr, harm in zip(
        data.loc_id[test_mask], label, pred_global, pred_semantic, accepted, global_rmse, smst_rmse, spectral_harm, strict=True
    ):
        rows.append({
            "direction": name,
            "loc_id": int(loc),
            "crop_label": str(crop),
            "global_prediction": str(gp),
            "semantic_prediction": str(sp),
            "smst_committed_semantic": bool(acc),
            "global_rmse": float(gr),
            "smst_rmse": float(sr),
            "spectral_harm_vs_global": bool(harm),
        })
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--r4-runner", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Output directory already exists")
    for name, expected in EXPECTED_INPUTS.items():
        actual = sha256(args.data_dir / name)
        if actual != expected:
            raise RuntimeError(f"Input hash mismatch for {name}: {actual}")
    r4 = load_r4(args.r4_runner.resolve())
    data = r4.load_data(args.data_dir.resolve())
    west = data.lon <= THRESHOLD_LONGITUDE - GAP_DEGREES
    east = data.lon >= THRESHOLD_LONGITUDE + GAP_DEGREES
    excluded = ~(west | east)
    west_east, rows_we = evaluate_direction(r4, data, west, east, "west_to_east")
    east_west, rows_ew = evaluate_direction(r4, data, east, west, "east_to_west")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = rows_we + rows_ew
    with (args.output / "R5_PER_LOCATION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "schema": "smst-r5-regional-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_POST_R4_SECONDARY_STRESS_TEST",
        "claim_boundary": "Same-release west/east geographic stress test after R4. It is not an independent dataset, a second sensor-pair evaluation, or a population-risk certificate.",
        "bindings": {
            "preregistration_sha256": sha256(args.prereg),
            "runner_sha256": sha256(Path(__file__)),
            "r4_runner_sha256": sha256(args.r4_runner),
            "inputs": {name: sha256(args.data_dir / name) for name in EXPECTED_INPUTS},
        },
        "fixed_configuration": {**CONFIG, "margin": MARGIN},
        "geography": {
            "longitude_threshold": THRESHOLD_LONGITUDE,
            "gap_degrees": GAP_DEGREES,
            "west_rows": int(west.sum()),
            "east_rows": int(east.sum()),
            "excluded_rows": int(excluded.sum()),
        },
        "directions": {"west_to_east": west_east, "east_to_west": east_west},
    }
    (args.output / "R5_SUMMARY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in sorted(args.output.iterdir()) if path.is_file()) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
