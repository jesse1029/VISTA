#!/usr/bin/env python3
"""Deterministic integrity audit for the R8 external-2020 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r8-runner", required=True, type=Path)
    parser.add_argument("--r4-runner", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--audit-out", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.outdir / "R8_EXTERNAL_SUMMARY.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(args.outdir / "R8_PER_LOCATION.csv")
    failures: list[str] = []
    if int(summary["external_pair"]["rows"]) != len(rows) or len(rows) != 1490:
        failures.append("external-row-count")
    if summary["input_sha256"]["r8_preregistration"] != sha256(args.prereg):
        failures.append("preregistration-hash")
    if summary["input_sha256"]["r8_runner"] != sha256(args.r8_runner):
        failures.append("r8-runner-hash")
    if summary["input_sha256"]["r4_runner"] != sha256(args.r4_runner):
        failures.append("r4-runner-hash")
    source = args.r8_runner.read_text(encoding="utf-8")
    if "core.indices(source, \"final\")" in source or "r4-final" in source.lower():
        failures.append("r4-final-access-pattern")
    selector_checks = {}
    for name, reported in summary["selectors_external"].items():
        accepted = rows[f"{name}_accept"].astype(bool).to_numpy()
        rmse = rows[f"{name}_rmse"].to_numpy(dtype=float)
        harm = rows[f"{name}_harm"].astype(bool).to_numpy()
        global_rmse = rows["global_rmse"].to_numpy(dtype=float)
        expected_harm = (rmse - global_rmse > 0.25) & ((rmse - global_rmse) / np.maximum(global_rmse, 1e-12) > 0.05)
        checks = {
            "coverage": float(accepted.mean()),
            "accepted_rows": int(accepted.sum()),
            "harm_count": int(harm.sum()),
            "harm_formula_match": bool(np.array_equal(harm, expected_harm)),
            "zero_label_change_rule": name.startswith("top_label") or name == "smst_top_label_and_raw_margin",
        }
        selector_checks[name] = checks
        for key, expected in (("coverage", checks["coverage"]), ("accepted_rows", checks["accepted_rows"]), ("declared_harms_vs_global", checks["harm_count"])):
            if abs(float(reported[key]) - float(expected)) > 1e-12:
                failures.append(f"selector-{name}-{key}")
        if not checks["harm_formula_match"]:
            failures.append(f"selector-{name}-harm-formula")
    checksums = {}
    for line in (args.outdir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.strip()
        actual = sha256(args.outdir / filename)
        checksums[filename] = actual == digest
        if not checksums[filename]:
            failures.append(f"checksum-{filename}")
    payload = {
        "schema": "smst-r8-external-2020-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "r8_runner_sha256": sha256(args.r8_runner),
        "r4_runner_sha256": sha256(args.r4_runner),
        "preregistration_sha256": sha256(args.prereg),
        "summary_sha256": sha256(args.outdir / "R8_EXTERNAL_SUMMARY.json"),
        "per_location_sha256": sha256(args.outdir / "R8_PER_LOCATION.csv"),
        "selector_checks": selector_checks,
        "checksum_matches": checksums,
        "claim_boundary": summary["claim_boundary"],
    }
    args.audit_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("R8 audit failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
