#!/usr/bin/env python3
"""Current five-fold paired-reference decoder anchor for VISTA TGRS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from run_common_utility_expanded import (
    CLASSES,
    PUBLISHED_LOCKS,
    apply_diag,
    apply_weighted_residual,
    fit_diag_affine,
    fit_logistic,
    fit_weighted_residual,
    load_release,
    make_partitions,
    sha256,
)


def summarize_group(frame: pd.DataFrame) -> dict:
    labels = frame["class_idx"].to_numpy(dtype=int)
    ref = frame["reference_pred"].to_numpy(dtype=int)
    fallback = frame["fallback_pred"].to_numpy(dtype=int)
    candidate = frame["candidate_pred"].to_numpy(dtype=int)
    returned = frame["returned_pred"].to_numpy(dtype=int)
    changed = candidate != fallback
    return {
        "n": int(len(frame)),
        "reference_accuracy": float(accuracy_score(labels, ref)),
        "reference_macro_f1": float(
            f1_score(labels, ref, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)
        ),
        "fallback_reference_agreement": float(np.mean(fallback == ref)),
        "returned_reference_agreement": float(np.mean(returned == ref)),
        "candidate_reference_agreement": float(np.mean(candidate == ref)),
        "candidate_label_changes": int(changed.sum()),
        "changes_toward_reference": int(np.sum(changed & (candidate == ref) & (fallback != ref))),
        "changes_away_from_reference": int(np.sum(changed & (fallback == ref) & (candidate != ref))),
        "changes_between_other_labels": int(
            np.sum(changed & (fallback != ref) & (candidate != ref))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desis", type=Path, required=True)
    parser.add_argument("--prisma", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_release(args.desis, args.prisma)
    parts = make_partitions(data["lon"], data["lat"])
    rows = []

    for part in parts:
        fold = int(part["fold"])
        dev = part["dev"]
        test = part["test"]
        lock = PUBLISHED_LOCKS[fold]
        global_state = fit_diag_affine(
            data["x"][dev], data["y"][dev], lock["stage1"]["global_alpha"]
        )
        decoder = fit_logistic(
            data["y"][dev], data["class_idx"][dev], lock["stage1"]["decoder_alpha"]
        )
        g_dev = apply_diag(data["x"][dev], global_state)
        g_test = apply_diag(data["x"][test], global_state)
        prob_dev = decoder.proba(g_dev)
        states = fit_weighted_residual(
            data["x"][dev],
            data["y"][dev],
            g_dev,
            prob_dev,
            lock["logistic_partial"]["residual_alpha"],
        )
        full_test = apply_weighted_residual(
            data["x"][test], g_test, decoder.proba(g_test), states
        )
        scale = float(lock["logistic_partial"]["scale"])
        candidate = g_test + scale * (full_test - g_test)

        reference_pred = decoder.predict_idx(data["y"][test])
        fallback_pred = decoder.predict_idx(g_test)
        candidate_pred = decoder.predict_idx(candidate)
        returned_pred = np.where(candidate_pred == fallback_pred, candidate_pred, fallback_pred)

        for pos, idx in enumerate(test):
            rows.append(
                {
                    "row_index": int(idx),
                    "loc_id": int(data["loc_id"][idx]),
                    "fold": fold,
                    "crop": str(data["crop"][idx]),
                    "class_idx": int(data["class_idx"][idx]),
                    "reference_pred": int(reference_pred[pos]),
                    "fallback_pred": int(fallback_pred[pos]),
                    "candidate_pred": int(candidate_pred[pos]),
                    "returned_pred": int(returned_pred[pos]),
                }
            )

    frame = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    if len(frame) != 16581 or frame["row_index"].nunique() != 16581:
        raise RuntimeError("Current anchor does not contain every row exactly once")

    cropwise = {str(crop): summarize_group(group) for crop, group in frame.groupby("crop")}
    summary = {
        "inputs": {
            "desis": str(args.desis.resolve()),
            "desis_sha256": sha256(args.desis),
            "prisma": str(args.prisma.resolve()),
            "prisma_sha256": sha256(args.prisma),
        },
        "protocol": "current five-fold published fallback/decoder and logistic partial-family locks",
        "overall": summarize_group(frame),
        "cropwise": cropwise,
    }
    frame.to_csv(args.out / "current_reference_anchor_per_location.csv", index=False)
    (args.out / "current_reference_anchor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
