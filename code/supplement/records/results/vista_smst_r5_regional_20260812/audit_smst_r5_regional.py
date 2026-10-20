#!/usr/bin/env python3
"""Independent integrity audit for the R5 regional stress-test output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--r4-runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.run / "R5_SUMMARY.json"
    rows_path = args.run / "R5_PER_LOCATION.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    bindings = summary.get("bindings", {})
    expected = {
        "preregistration_sha256": sha256(args.prereg),
        "runner_sha256": sha256(args.runner),
        "r4_runner_sha256": sha256(args.r4_runner),
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            failures.append(f"binding mismatch: {key}")
    rows = list(csv.DictReader(rows_path.open(newline="", encoding="utf-8")))
    if len(rows) != 15407:
        failures.append(f"unexpected per-location row count: {len(rows)}")
    by_direction: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_direction.setdefault(row["direction"], []).append(row)
        if row["smst_committed_semantic"] == "True" and row["semantic_prediction"] != row["global_prediction"]:
            failures.append("a committed semantic row changes the frozen top label")
            break
    for name, expected_test in (("west_to_east", 7765), ("east_to_west", 7642)):
        group = by_direction.get(name, [])
        if len(group) != expected_test:
            failures.append(f"{name} test count mismatch")
            continue
        reported = summary["directions"][name]["paired_vs_global"]
        coverage = sum(r["smst_committed_semantic"] == "True" for r in group) / len(group)
        harms = sum(r["spectral_harm_vs_global"] == "True" for r in group)
        if abs(coverage - float(reported["coverage"])) > 1e-12:
            failures.append(f"{name} coverage mismatch")
        if harms != int(reported["spectral_harms_vs_global"]):
            failures.append(f"{name} spectral-harm mismatch")
        if int(reported["semantic_label_changes_after_gate"]) != 0:
            failures.append(f"{name} post-gate label-change count is nonzero")
    payload = {
        "schema": "smst-r5-regional-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "bindings": {
            "summary": sha256(summary_path),
            "per_location": sha256(rows_path),
            **expected,
        },
        "boundary": "The audit validates bindings and decoder-relative accounting for a same-release post-R4 regional stress test. It does not make the test independent or certify population risk.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
