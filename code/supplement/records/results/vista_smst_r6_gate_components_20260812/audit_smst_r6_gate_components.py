#!/usr/bin/env python3
"""Integrity audit for the fixed R6 gate-component characterization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise RuntimeError(f"Invalid boolean value: {value}")
    return value == "True"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--r4-summary", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "R6_SUMMARY.json").read_text(encoding="utf-8"))
    r4 = json.loads(args.r4_summary.read_text(encoding="utf-8"))
    rows = list(csv.DictReader((args.run / "R6_PER_LOCATION.csv").open(encoding="utf-8")))
    errors: list[str] = []
    if summary.get("schema") != "smst-r6-fixed-gate-components-v1": errors.append("schema")
    if summary.get("final_rows") != len(rows): errors.append("row_count")
    if summary.get("r4_runner_sha256") != "3a14e29f063f5809d941f4b202483bb017345475164c6c10a6eba90cffd770f1": errors.append("r4_runner_hash")
    if summary.get("r4_preregistration_sha256") != "7618c6c5859cf43e84a7a158121c02872bb519c8d548331012d1c1c6a18ae04a": errors.append("r4_prereg_hash")
    if summary.get("r4_decision_lock_sha256") != r4.get("decision_lock_sha256"): errors.append("decision_lock")
    if summary.get("variants", {}).get("both", {}).get("coverage") != r4.get("paired_vs_global", {}).get("coverage"): errors.append("both_coverage")
    if summary.get("variants", {}).get("both", {}).get("delta_rmse_vs_global") != r4.get("paired_vs_global", {}).get("smst_minus_global_rmse"): errors.append("both_rmse")
    if summary.get("variants", {}).get("both", {}).get("frozen_decoder_label_changes_vs_global") != 0: errors.append("both_labels")
    counts = {name: 0 for name in ("unconstrained", "label_only", "margin_only", "both")}
    violations = {"both_not_intersection": 0, "label_changes_in_label_only": 0}
    for row in rows:
        flags = {name: as_bool(row[f"accept_{name}"]) for name in counts}
        for name in counts: counts[name] += int(flags[name])
        label = row["semantic_prediction"] == row["global_prediction"]
        margin = float(row["semantic_margin"]) >= float(row["global_margin"])
        violations["both_not_intersection"] += int(flags["both"] != (label and margin))
        violations["label_changes_in_label_only"] += int(flags["label_only"] and not label)
    if counts["unconstrained"] != len(rows): errors.append("unconstrained_coverage")
    if any(violations.values()): errors.append("predicate_consistency")
    payload = {
        "schema": "smst-r6-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "input_sha256": {"runner": sha256(args.runner), "preregistration": sha256(args.prereg), "r4_summary": sha256(args.r4_summary), "summary": sha256(args.run / "R6_SUMMARY.json"), "per_location": sha256(args.run / "R6_PER_LOCATION.csv")},
        "row_count": len(rows), "accepted_rows": counts, "predicate_violations": violations,
        "boundary": "Integrity audit only; it neither refits R4 nor turns R6 into an independent or primary evaluation.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
