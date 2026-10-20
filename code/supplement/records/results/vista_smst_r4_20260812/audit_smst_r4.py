#!/usr/bin/env python3
"""Independent, read-only integrity audit for the completed SMST-R4 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_INPUTS = {
    "desis_2021.csv": "85ac21e38fb990a75714f90be0d2ced643275adf4e46570f326e765709c48ccc",
    "prisma_2021.csv": "0c6ef3dff0e5eff1c3e9ebb349494465d4459ee189c2aeb089af4e7622822647",
    "desis_prisma_2021_paper_reconciled.xml": "1453cf193438e59ad61d4cdde2eb4c089c1c28c843517585f7eb47c0f4e5aa46",
}
SEED_TEXT = "smst-r4-20260812:"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bucket(block: str) -> int:
    return int.from_bytes(hashlib.sha256((SEED_TEXT + block).encode()).digest()[:8], "big") % 10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--prereg", required=True, type=Path)
    p.add_argument("--runner", required=True, type=Path)
    p.add_argument("--development", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--final", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    failures: list[str] = []
    for name, expected in EXPECTED_INPUTS.items():
        if sha256(a.data_dir / name) != expected:
            failures.append(f"input_hash:{name}")
    summary = json.loads((a.final / "FINAL_SUMMARY.json").read_text(encoding="utf-8"))
    decision = json.loads((a.validation / "FINAL_DECISION_LOCK.json").read_text(encoding="utf-8"))
    dev = json.loads((a.development / "DEVELOPMENT_MANIFEST.json").read_text(encoding="utf-8"))
    if summary["pre_registration_sha256"] != sha256(a.prereg): failures.append("prereg_binding")
    if summary["script_sha256"] != sha256(a.runner): failures.append("runner_binding")
    if summary["decision_lock_sha256"] != sha256(a.validation / "FINAL_DECISION_LOCK.json"): failures.append("decision_lock_binding")
    if decision["development_models_sha256"] != sha256(a.development / "development_models.pkl"): failures.append("development_model_binding")
    if summary["selected"] != decision["selected"]: failures.append("selected_config_binding")

    raw = pd.read_csv(a.data_dir / "desis_2021.csv", usecols=["loc_id", "longitude", "latitude"])
    blocks = [f"{int(np.floor((x + 180.0) / .01))}:{int(np.floor((y + 90.0) / .01))}" for x, y in zip(raw.longitude, raw.latitude, strict=True)]
    expected_final = sorted(raw.loc_id[np.asarray([bucket(v) >= 8 for v in blocks])].astype(int).tolist())
    expected_final_hash = hashlib.sha256("\n".join(map(str, expected_final)).encode()).hexdigest()
    if dev["split"]["loc_id_list_sha256"]["final"] != expected_final_hash: failures.append("split_manifest_rebuild")

    per = pd.read_csv(a.final / "final_per_location.csv")
    if per.loc_id.duplicated().any(): failures.append("duplicate_final_loc_id")
    if sorted(per.loc_id.astype(int).tolist()) != expected_final: failures.append("final_loc_id_set")
    if len(per) != summary["final_row_count"]: failures.append("final_row_count")
    committed = per.smst_committed_semantic.astype(bool).to_numpy()
    semantic_equal_when_committed = (per.semantic_prediction.to_numpy()[committed] == per.global_prediction.to_numpy()[committed]).all()
    if not semantic_equal_when_committed: failures.append("committed_label_preservation")
    coverage = float(committed.mean())
    if not np.isclose(coverage, summary["paired_vs_global"]["coverage"], atol=1e-12): failures.append("coverage")
    delta = per.smst_rmse.to_numpy() - per.global_rmse.to_numpy()
    spectral_harm = (delta > 0.25) & (delta / np.maximum(per.global_rmse.to_numpy(), 1e-12) > 0.05)
    if not np.array_equal(spectral_harm, per.spectral_harm_vs_global.astype(bool).to_numpy()): failures.append("spectral_harm_rows")
    if int(spectral_harm.sum()) != summary["paired_vs_global"]["spectral_harms_vs_global"]: failures.append("spectral_harm_count")
    if int(per.union_harm_vs_global.astype(bool).sum()) != summary["paired_vs_global"]["component_union_harms_vs_global"]: failures.append("union_harm_count")
    report = {
        "schema": "smst-r4-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "bindings": {"prereg": sha256(a.prereg), "runner": sha256(a.runner), "development_models": sha256(a.development / "development_models.pkl"), "decision": sha256(a.validation / "FINAL_DECISION_LOCK.json"), "final_summary": sha256(a.final / "FINAL_SUMMARY.json"), "final_per_location": sha256(a.final / "final_per_location.csv")},
        "checks": {"final_rows": int(len(per)), "coverage": coverage, "committed_label_preservation": bool(semantic_equal_when_committed), "spectral_harms": int(spectral_harm.sum()), "union_harms": int(per.union_harm_vs_global.astype(bool).sum())},
        "boundary": "PASS validates experiment integrity and the frozen-decoder top-label gate. It does not certify true material state, population risk, or external generalization.",
    }
    (a.output / "AUDIT_SMST_R4.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (a.output / "SHA256SUMS").write_text(f"{sha256(a.output / 'AUDIT_SMST_R4.json')}  AUDIT_SMST_R4.json\n", encoding="utf-8")
    if failures:
        raise SystemExit("SMST-R4 audit failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
