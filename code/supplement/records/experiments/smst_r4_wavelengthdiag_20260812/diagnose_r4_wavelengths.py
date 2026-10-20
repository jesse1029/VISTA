#!/usr/bin/env python3
"""Post-R4 descriptive wavelength diagnostics for the locked selected model.

This script never fits or selects a model.  It reconstructs the locked R4
final outputs from the immutable development state and records band-wise
paired-error summaries.  The raw release remains on the remote host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run_smst_r4 import indices, load_data, logits, top_and_margin, transports


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--per-location", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {args.output}")
    data = load_data(args.data_dir)
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    with args.development.open("rb") as handle:
        models = pickle.load(handle)
    selected = decision["selected"]
    bundle = models[selected["model_key"]]
    final = indices(data, "final")
    z, y = data.z_prisma[final], data.y_desis[final]
    global_x, semantic_x, global_idx, margin_global, margin_semantic = transports(bundle, z)
    semantic_idx, _ = top_and_margin(logits(bundle, semantic_x))
    accepted = (semantic_idx == global_idx) & (margin_semantic >= margin_global + float(selected["margin"]))
    smst_x = np.where(accepted[:, None], semantic_x, global_x)

    archived = pd.read_csv(args.per_location)
    if len(archived) != len(final) or not np.array_equal(archived.loc_id.to_numpy(np.int64), data.loc_id[final]):
        raise RuntimeError("Archived final per-location order differs from the locked final split")
    archived_acceptance = archived.smst_committed_semantic.to_numpy(bool)
    if not np.array_equal(archived_acceptance, accepted):
        raise RuntimeError("Reconstructed gate does not reproduce archived final acceptance")
    archived_global = archived.global_rmse.to_numpy(float)
    archived_smst = archived.smst_rmse.to_numpy(float)
    reconstructed_global = np.sqrt(np.mean((global_x - y) ** 2, axis=1))
    reconstructed_smst = np.sqrt(np.mean((smst_x - y) ** 2, axis=1))
    if not np.allclose(archived_global, reconstructed_global, rtol=0.0, atol=1e-12):
        raise RuntimeError("Reconstructed global RMSE differs from archived final record")
    if not np.allclose(archived_smst, reconstructed_smst, rtol=0.0, atol=1e-12):
        raise RuntimeError("Reconstructed SMST RMSE differs from archived final record")

    global_error = global_x - y
    smst_error = smst_x - y
    args.output.mkdir(parents=True)
    csv_path = args.output / "R4_WAVELENGTH_DIAGNOSTICS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "wavelength_nm",
            "global_bias",
            "global_rmse",
            "smst_bias",
            "smst_rmse",
            "smst_minus_global_rmse",
            "accepted_semantic_rmse",
            "accepted_semantic_bias",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, wavelength in enumerate(data.desis_wavelength_nm):
            residual_band = semantic_x[accepted, index] - y[accepted, index]
            writer.writerow(
                {
                    "wavelength_nm": float(wavelength),
                    "global_bias": float(global_error[:, index].mean()),
                    "global_rmse": float(np.sqrt(np.mean(global_error[:, index] ** 2))),
                    "smst_bias": float(smst_error[:, index].mean()),
                    "smst_rmse": float(np.sqrt(np.mean(smst_error[:, index] ** 2))),
                    "smst_minus_global_rmse": float(np.sqrt(np.mean(smst_error[:, index] ** 2)) - np.sqrt(np.mean(global_error[:, index] ** 2))),
                    "accepted_semantic_rmse": float(np.sqrt(np.mean(residual_band ** 2))),
                    "accepted_semantic_bias": float(residual_band.mean()),
                }
            )
    summary = {
        "schema": "smst-r4-posthoc-wavelength-diagnostics-v1",
        "scope": "Post-R4 descriptive reconstruction of the locked final outputs; no refit, reselection, or additional primary test.",
        "model_key": selected["model_key"],
        "margin": float(selected["margin"]),
        "final_rows": int(len(final)),
        "accepted_rows": int(accepted.sum()),
        "wavelength_count": int(len(data.desis_wavelength_nm)),
        "wavelength_min_nm": float(data.desis_wavelength_nm.min()),
        "wavelength_max_nm": float(data.desis_wavelength_nm.max()),
        "max_reconstruction_rmse_error": {
            "global": float(np.max(np.abs(archived_global - reconstructed_global))),
            "smst": float(np.max(np.abs(archived_smst - reconstructed_smst))),
        },
        "inputs_sha256": {
            "development_models.pkl": sha256(args.development),
            "FINAL_DECISION_LOCK.json": sha256(args.decision),
            "final_per_location.csv": sha256(args.per_location),
            "runner": sha256(Path(__file__).with_name("run_smst_r4.py")),
            "diagnostic_script": sha256(Path(__file__)),
        },
        "diagnostic_csv_sha256": sha256(csv_path),
    }
    write_json(args.output / "R4_WAVELENGTH_DIAGNOSTICS_SUMMARY.json", summary)
    manifest_paths = [csv_path, args.output / "R4_WAVELENGTH_DIAGNOSTICS_SUMMARY.json"]
    (args.output / "SHA256SUMS").write_text("\n".join(f"{sha256(path)}  {path.name}" for path in manifest_paths) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
