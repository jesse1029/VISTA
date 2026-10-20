#!/usr/bin/env python3
"""Post-review matched-comparator and spatial-bootstrap analysis for SMST.

The script freezes every new candidate definition and every hyperparameter grid
before touching final outcomes.  It imports (and hash-checks) the original R4
runner, preserves its spatial split, and treats the final partition as a
single retrospective secondary evaluation.  It is deliberately explicit about
the scope: this analysis tests attribution on the same paired 2021 release; it
does not provide an independent sensor, season, or population evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


R4_RUNNER_SHA256 = "3a14e29f063f5809d941f4b202483bb017345475164c6c10a6eba90cffd770f1"
R4_PREREG_SHA256 = "7618c6c5859cf43e84a7a158121c02872bb519c8d548331012d1c1c6a18ae04a"
SMST_KEY = "g=1e-02|d=1e-01|r=1e-04"
HARM_ABSOLUTE = 0.25
HARM_RELATIVE = 0.05
SELECTION_GRID = {
    "diagonal_affine_alpha": [1e-6, 1e-4, 1e-2, 1.0],
    "multivariate_ridge_alpha": [1e-4, 1e-2, 1.0, 100.0],
    "pls_components": [2, 5, 10, 20],
    "pds_window": [3, 7, 11],
    "residual_alpha": [1e-6, 1e-4, 1e-2, 1.0],
    "cluster_count": [3, 5, 7],
}
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260812
CURVE_QUANTILES = 21


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def per_sample(core: Any, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rmse = np.sqrt(np.mean((x - y) ** 2, axis=1))
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    cosine = np.divide(np.sum(x * y, axis=1), denom, out=np.ones_like(denom), where=denom > 0)
    sam = np.arccos(np.clip(cosine, -1.0, 1.0))
    return rmse, sam


def top_and_margin(core: Any, bundle: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return core.top_and_margin(core.logits(bundle, x))


def fit_diagonal(core: Any, z: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    return core.fit_diagonal_affine(z, y, alpha)


def apply_diagonal(core: Any, z: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return core.apply_affine(z, model[0], model[1])


def fit_ridge(z: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    return Ridge(alpha=alpha, fit_intercept=True).fit(z, y)


def fit_pls(z: np.ndarray, y: np.ndarray, components: int) -> PLSRegression:
    return PLSRegression(n_components=components, scale=True, max_iter=1000, tol=1e-8).fit(z, y)


def fit_pds(z: np.ndarray, y: np.ndarray, window: int) -> list[Ridge]:
    half = window // 2
    models: list[Ridge] = []
    for band in range(y.shape[1]):
        lo, hi = max(0, band - half), min(z.shape[1], band + half + 1)
        models.append(Ridge(alpha=1e-2, fit_intercept=True).fit(z[:, lo:hi], y[:, band]))
    return models


def predict_pds(z: np.ndarray, models: list[Ridge]) -> np.ndarray:
    output = np.empty((z.shape[0], len(models)), dtype=np.float64)
    for band, model in enumerate(models):
        half = (model.n_features_in_ - 1) // 2
        lo = max(0, band - half)
        hi = lo + model.n_features_in_
        if hi > z.shape[1]:
            hi = z.shape[1]
            lo = hi - model.n_features_in_
        output[:, band] = model.predict(z[:, lo:hi])
    return output


def fit_residual_diagonal(core: Any, z: np.ndarray, target: np.ndarray, base: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    return core.fit_diagonal_affine(z, target - base, alpha)


def fit_class_residuals(
    core: Any,
    z: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    residual = target - base
    models: list[tuple[np.ndarray, np.ndarray]] = []
    for class_idx in range(weights.shape[1]):
        models.append(core.fit_diagonal_affine(z, residual, alpha, weights[:, class_idx]))
    return models


def predict_soft_residual(core: Any, z: np.ndarray, base: np.ndarray, models: list[tuple[np.ndarray, np.ndarray]], weights: np.ndarray) -> np.ndarray:
    residual = np.stack([core.apply_affine(z, item[0], item[1]) for item in models], axis=1)
    return base + np.einsum("nc,ncd->nd", weights, residual)


def predict_hard_residual(core: Any, z: np.ndarray, base: np.ndarray, models: list[tuple[np.ndarray, np.ndarray]], index: np.ndarray) -> np.ndarray:
    residual = np.stack([core.apply_affine(z, item[0], item[1]) for item in models], axis=1)
    return base + residual[np.arange(z.shape[0]), index]


def choose(
    name: str,
    candidates: list[tuple[dict[str, Any], Callable[[], tuple[Callable[[np.ndarray], np.ndarray], Any]]]],
    z_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[dict[str, Any], Callable[[np.ndarray], np.ndarray], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best: tuple[float, str, dict[str, Any], Callable[[np.ndarray], np.ndarray]] | None = None
    for params, factory in candidates:
        predictor, _model = factory()
        prediction = predictor(z_validation)
        value = float(np.sqrt(np.mean((prediction - y_validation) ** 2, axis=1)).mean())
        parameter_key = json.dumps(params, sort_keys=True, separators=(",", ":"))
        rows.append({"candidate": name, "params": params, "validation_rmse": value})
        item = (value, parameter_key, params, predictor)
        if best is None or item[:2] < best[:2]:
            best = item
    if best is None:
        raise RuntimeError(f"No validation candidates for {name}")
    return best[2], best[3], rows


def cell_ids(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{int(np.floor((x + 180.0) / .01))}:{int(np.floor((y + 90.0) / .01))}" for x, y in zip(lon, lat, strict=True)],
        dtype=str,
    )


def harmonic_metric(labels: np.ndarray, values: np.ndarray, classes: np.ndarray) -> float:
    return float(np.mean([values[labels == c].mean() for c in classes]))


def paired_summary(
    core: Any,
    labels: np.ndarray,
    classes: np.ndarray,
    global_x: np.ndarray,
    candidate_x: np.ndarray,
    global_pred: np.ndarray,
    candidate_pred: np.ndarray,
    accepted: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    committed_x = np.where(accepted[:, None], candidate_x, global_x)
    committed_pred = np.where(accepted, candidate_pred, global_pred)
    global_rmse, global_sam = per_sample(core, global_x, target)
    committed_rmse, committed_sam = per_sample(core, committed_x, target)
    excess = committed_rmse - global_rmse
    harm = (excess > HARM_ABSOLUTE) & (excess / np.maximum(global_rmse, 1e-12) > HARM_RELATIVE)
    c2i = (global_pred == labels) & (committed_pred != labels)
    metric = core.metrics(committed_pred, labels, committed_x, target, classes)
    for key in ("per_sample_rmse", "per_sample_sam"):
        metric.pop(key, None)
    global_metric = core.metrics(global_pred, labels, global_x, target, classes)
    for key in ("per_sample_rmse", "per_sample_sam"):
        global_metric.pop(key, None)
    summary = {
        "coverage": float(accepted.mean()),
        "commits": int(accepted.sum()),
        "top_label_changes_before_commit": int((candidate_pred != global_pred).sum()),
        "top_label_changes_after_commit": int((committed_pred != global_pred).sum()),
        "rmse": float(metric["rmse"]),
        "sam_rad": float(metric["sam_rad"]),
        "macro_f1": float(metric["macro_f1"]),
        "delta_rmse_vs_global": float(metric["rmse"] - global_metric["rmse"]),
        "delta_sam_rad_vs_global": float(metric["sam_rad"] - global_metric["sam_rad"]),
        "delta_macro_f1_vs_global": float(metric["macro_f1"] - global_metric["macro_f1"]),
        "relative_rmse_reduction_pct": float(100.0 * (global_metric["rmse"] - metric["rmse"]) / global_metric["rmse"]),
        "declared_harms_all": int(harm.sum()),
        "declared_harm_rate_all": float(harm.mean()),
        "declared_harms_commits": int((harm & accepted).sum()),
        "declared_harm_rate_commits": float((harm & accepted).sum() / max(1, accepted.sum())),
        "correct_to_incorrect": int(c2i.sum()),
        "union_harms_all": int((harm | c2i).sum()),
        "union_harm_rate_all": float((harm | c2i).mean()),
        "mean_positive_rmse_excess": float(np.maximum(excess, 0.0).mean()),
        "cvar95_rmse_excess": float(np.mean(np.sort(excess)[max(0, int(0.95 * excess.size) - 1):])),
    }
    arrays = {
        "global_rmse": global_rmse,
        "global_sam": global_sam,
        "committed_rmse": committed_rmse,
        "committed_sam": committed_sam,
        "harm": harm,
        "accepted": accepted,
        "c2i": c2i,
        "excess": excess,
    }
    return summary, arrays


def bootstrap_intervals(
    labels: np.ndarray,
    classes: np.ndarray,
    blocks: np.ndarray,
    smst: dict[str, np.ndarray],
    label_only: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unique, inverse = np.unique(blocks, return_inverse=True)
    by_block = [np.flatnonzero(inverse == idx) for idx in range(unique.size)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fields = [
        "delta_rmse_smst_vs_global",
        "delta_sam_smst_vs_global",
        "delta_balanced_rmse_smst_vs_global",
        "smst_any_rmse_worsening_rate",
        "smst_declared_harm_rate_all",
        "smst_declared_harm_rate_commits",
        "margin_minus_label_only_rmse",
        "margin_minus_label_only_commit_harm_rate",
    ]
    values = {field: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for field in fields}
    for rep in range(BOOTSTRAP_REPLICATES):
        picked_blocks = rng.integers(0, unique.size, size=unique.size)
        index = np.concatenate([by_block[item] for item in picked_blocks])
        delta = smst["committed_rmse"][index] - smst["global_rmse"][index]
        delta_sam = smst["committed_sam"][index] - smst["global_sam"][index]
        values["delta_rmse_smst_vs_global"][rep] = delta.mean()
        values["delta_sam_smst_vs_global"][rep] = delta_sam.mean()
        values["delta_balanced_rmse_smst_vs_global"][rep] = harmonic_metric(labels[index], delta, classes)
        values["smst_any_rmse_worsening_rate"][rep] = float((delta > 0).mean())
        values["smst_declared_harm_rate_all"][rep] = float(smst["harm"][index].mean())
        accepted = smst["accepted"][index]
        values["smst_declared_harm_rate_commits"][rep] = float((smst["harm"][index] & accepted).sum() / max(1, accepted.sum()))
        values["margin_minus_label_only_rmse"][rep] = float(smst["committed_rmse"][index].mean() - label_only["committed_rmse"][index].mean())
        label_accept = label_only["accepted"][index]
        smst_rate = float((smst["harm"][index] & accepted).sum() / max(1, accepted.sum()))
        label_rate = float((label_only["harm"][index] & label_accept).sum() / max(1, label_accept.sum()))
        values["margin_minus_label_only_commit_harm_rate"][rep] = smst_rate - label_rate
    summary = {
        "schema": "smst-review-e2-spatial-cell-bootstrap-v1",
        "spatial_block_rule": "floor((longitude+180)/0.01):floor((latitude+90)/0.01)",
        "unique_final_cells": int(unique.size),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "statistics": {
            name: {
                "mean": float(data.mean()),
                "ci95_low": float(np.quantile(data, 0.025)),
                "ci95_high": float(np.quantile(data, 0.975)),
            }
            for name, data in values.items()
        },
    }
    rows = [{"replicate": rep, **{name: float(data[rep]) for name, data in values.items()}} for rep in range(BOOTSTRAP_REPLICATES)]
    return summary, rows


def stable_subset(locations: np.ndarray, compatible: np.ndarray, count: int) -> np.ndarray:
    scores = np.asarray(
        [int.from_bytes(hashlib.sha256(f"smst-review-e3-matched-coverage-20260812:{int(value)}".encode()).digest()[:8], "big") for value in locations],
        dtype=np.uint64,
    )
    selected = np.zeros(locations.shape[0], dtype=bool)
    candidate = np.flatnonzero(compatible)
    selected[candidate[np.argsort(scores[candidate], kind="stable")[:count]]] = True
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4-runner", required=True, type=Path)
    parser.add_argument("--r4-prereg", required=True, type=Path)
    parser.add_argument("--r4-development", required=True, type=Path)
    parser.add_argument("--r4-validation", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    if args.outdir.exists():
        raise RuntimeError(f"Refusing to overwrite output: {args.outdir}")
    if sha256(args.r4_runner) != R4_RUNNER_SHA256 or sha256(args.r4_prereg) != R4_PREREG_SHA256:
        raise RuntimeError("R4 authority hash mismatch")
    sys.path.insert(0, str(args.r4_runner.parent))
    import run_smst_r4 as core  # noqa: PLC0415

    data = core.load_data(args.data_dir.resolve())
    development, validation, final = (core.indices(data, name) for name in ("development", "validation", "final"))
    z_dev, y_dev, labels_dev = data.z_prisma[development], data.y_desis[development], data.crop[development]
    z_val, y_val = data.z_prisma[validation], data.y_desis[validation]

    with (args.r4_development / "development_models.pkl").open("rb") as handle:
        r4_bundle = pickle.load(handle)[SMST_KEY]
    decision = json.loads((args.r4_validation / "FINAL_DECISION_LOCK.json").read_text(encoding="utf-8"))
    if decision["selected"]["model_key"] != SMST_KEY:
        raise RuntimeError("Unexpected R4 selected model")

    # All new candidates are selected on development/validation only.
    selection_rows: list[dict[str, Any]] = []
    diagonal_params, diagonal, rows = choose(
        "diagonal_affine",
        [
            ({"alpha": alpha}, lambda alpha=alpha: (lambda x, m=fit_diagonal(core, z_dev, y_dev, alpha): apply_diagonal(core, x, m), None))
            for alpha in SELECTION_GRID["diagonal_affine_alpha"]
        ],
        z_val,
        y_val,
    )
    selection_rows.extend(rows)
    # All conditional candidates use the frozen R4 global fallback as their
    # common base.  The separately selected diagonal map remains a matched
    # transport comparator rather than silently replacing that fallback.
    global_dev = core.apply_affine(z_dev, r4_bundle["global_a"], r4_bundle["global_b"])
    global_val = core.apply_affine(z_val, r4_bundle["global_a"], r4_bundle["global_b"])

    ridge_params, ridge_predictor, rows = choose(
        "multivariate_ridge",
        [
            ({"alpha": alpha}, lambda alpha=alpha: (lambda x, m=fit_ridge(z_dev, y_dev, alpha): m.predict(x), None))
            for alpha in SELECTION_GRID["multivariate_ridge_alpha"]
        ],
        z_val,
        y_val,
    )
    selection_rows.extend(rows)
    pls_params, pls_predictor, rows = choose(
        "pls",
        [
            ({"components": count}, lambda count=count: (lambda x, m=fit_pls(z_dev, y_dev, count): np.asarray(m.predict(x), dtype=np.float64), None))
            for count in SELECTION_GRID["pls_components"]
        ],
        z_val,
        y_val,
    )
    selection_rows.extend(rows)
    pds_params, pds_predictor, rows = choose(
        "pds",
        [
            ({"window": window}, lambda window=window: (lambda x, m=fit_pds(z_dev, y_dev, window): predict_pds(x, m), None))
            for window in SELECTION_GRID["pds_window"]
        ],
        z_val,
        y_val,
    )
    selection_rows.extend(rows)

    unconditional_params, unconditional_predictor, rows = choose(
        "unconditional_residual",
        [
            ({"alpha": alpha}, lambda alpha=alpha: (lambda x, m=fit_residual_diagonal(core, z_dev, y_dev, global_dev, alpha): core.apply_affine(x, r4_bundle["global_a"], r4_bundle["global_b"]) + apply_diagonal(core, x, m), None))
            for alpha in SELECTION_GRID["residual_alpha"]
        ],
        z_val,
        y_val,
    )
    selection_rows.extend(rows)

    # Decoder pseudo-class scores are trained only on development DESIS spectra.
    dev_prob = r4_bundle["decoder"].predict_proba(r4_bundle["scaler"].transform(global_dev))
    dev_hard = np.argmax(dev_prob, axis=1)
    hard_params, hard_predictor, rows = choose(
        "hard_predicted_class_residual",
        [
            ({"alpha": alpha}, lambda alpha=alpha: (lambda x, models=fit_class_residuals(core, z_dev, y_dev, global_dev, np.eye(len(data.classes))[dev_hard], alpha): predict_hard_residual(core, x, core.apply_affine(x, r4_bundle["global_a"], r4_bundle["global_b"]), models, np.argmax(r4_bundle["decoder"].predict_proba(r4_bundle["scaler"].transform(core.apply_affine(x, r4_bundle["global_a"], r4_bundle["global_b"]))), axis=1)), None))
            for alpha in SELECTION_GRID["residual_alpha"]
        ],
        z_val,
        y_val,
    )
    selection_rows.extend(rows)

    soft_params, soft_predictor, rows = choose(
        "soft_decoder_weighted_residual",
        [
            ({"alpha": alpha}, lambda alpha=alpha: (lambda x, models=fit_class_residuals(core, z_dev, y_dev, global_dev, dev_prob, alpha): predict_soft_residual(core, x, core.apply_affine(x, r4_bundle["global_a"], r4_bundle["global_b"]), models, r4_bundle["decoder"].predict_proba(r4_bundle["scaler"].transform(core.apply_affine(x, r4_bundle["global_a"], r4_bundle["global_b"])))), None))
            for alpha in SELECTION_GRID["residual_alpha"]
        ],
        z_val,
        y_val,
    )
    selection_rows.extend(rows)

    # Cluster models are fit without crop labels. The assigned residual map is deterministic.
    cluster_candidates: list[tuple[dict[str, Any], Callable[[], tuple[Callable[[np.ndarray], np.ndarray], Any]]]] = []
    for clusters in SELECTION_GRID["cluster_count"]:
        for alpha in SELECTION_GRID["residual_alpha"]:
            def factory(clusters=clusters, alpha=alpha):
                clusterer = KMeans(n_clusters=clusters, n_init=20, random_state=20260812).fit(global_dev)
                indices = clusterer.predict(global_dev)
                one_hot = np.eye(clusters)[indices]
                models = fit_class_residuals(core, z_dev, y_dev, global_dev, one_hot, alpha)
                def predictor(x: np.ndarray) -> np.ndarray:
                    base = core.apply_affine(x, r4_bundle["global_a"], r4_bundle["global_b"])
                    assignment = clusterer.predict(base)
                    return predict_hard_residual(core, x, base, models, assignment)
                return predictor, (clusterer, models)
            cluster_candidates.append(({"clusters": clusters, "alpha": alpha}, factory))
    cluster_params, cluster_predictor, rows = choose("unsupervised_cluster_residual", cluster_candidates, z_val, y_val)
    selection_rows.extend(rows)

    # The original R4 soft candidate and its development/validation lock are retained as the named SMST candidate.
    def r4_soft(x: np.ndarray) -> np.ndarray:
        global_x, semantic_x, _idx, _mg, _ms = core.transports(r4_bundle, x)
        del global_x
        return semantic_x

    # Diagnostic oracle weights residual maps by supplied crop labels and is not deployed.
    r4_dev_prob = r4_bundle["decoder"].predict_proba(r4_bundle["scaler"].transform(global_dev))
    oracle_models = fit_class_residuals(core, z_dev, y_dev, global_dev, r4_dev_prob, float(r4_bundle["config"]["residual_alpha"]))

    args.outdir.mkdir(parents=True)
    lock = {
        "schema": "smst-review-e1e3-decision-lock-v1",
        "claim_boundary": "Post-review secondary analysis on the previously opened R4 split. New model choices use development/validation only; final statistics are not independent confirmation.",
        "r4_runner_sha256": sha256(args.r4_runner),
        "r4_preregistration_sha256": sha256(args.r4_prereg),
        "r4_decision_lock_sha256": sha256(args.r4_validation / "FINAL_DECISION_LOCK.json"),
        "input_hashes": core.EXPECTED_INPUTS,
        "selection_rule": "minimum validation spectral RMSE; ties resolved by canonical JSON parameter string",
        "selection_grid": SELECTION_GRID,
        "selected": {
            "diagonal_affine": diagonal_params,
            "multivariate_ridge": ridge_params,
            "pls": pls_params,
            "pds": pds_params,
            "unconditional_residual": unconditional_params,
            "unsupervised_cluster_residual": cluster_params,
            "hard_predicted_class_residual": hard_params,
            "soft_decoder_weighted_residual": soft_params,
            "r4_smt_candidate": {"model_key": SMST_KEY, "margin": decision["selected"]["margin"]},
        },
        "margin_curve": {
            "top_label_compatibility": "candidate top class equals frozen decoder top class on global fallback",
            "thresholds": "nonnegative empirical quantiles of validation candidate-minus-global logit margin, including zero",
            "matched_coverage_sampling": "fixed SHA-256 ordering of final locations within the top-label-compatible set",
        },
        "bootstrap": {
            "spatial_block_rule": "floor((longitude+180)/0.01):floor((latitude+90)/0.01)",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "final_evaluation_opened": False,
    }
    dump(args.outdir / "DECISION_LOCK.json", lock)
    with (args.outdir / "VALIDATION_SELECTION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", "params", "validation_rmse"])
        writer.writeheader()
        for row in selection_rows:
            writer.writerow({"candidate": row["candidate"], "params": json.dumps(row["params"], sort_keys=True), "validation_rmse": row["validation_rmse"]})

    # Only after the lock is durable do we compute final values.
    z_final, y_final, labels_final = data.z_prisma[final], data.y_desis[final], data.crop[final]
    global_x = core.apply_affine(z_final, r4_bundle["global_a"], r4_bundle["global_b"])
    global_idx, global_margin = top_and_margin(core, r4_bundle, global_x)
    global_pred = data.classes[global_idx]
    predictors: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str]] = {
        "diagonal_affine": (diagonal, "transport"),
        "multivariate_ridge": (ridge_predictor, "transport"),
        "pls": (pls_predictor, "transport"),
        "pds": (pds_predictor, "transport"),
        "unconditional_residual": (unconditional_predictor, "residual"),
        "unsupervised_cluster_residual": (cluster_predictor, "residual"),
        "hard_predicted_class_residual": (hard_predictor, "residual"),
        "soft_decoder_weighted_residual": (soft_predictor, "residual"),
        "r4_soft_decoder_weighted": (r4_soft, "residual"),
    }
    all_rows: list[dict[str, Any]] = []
    detailed: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for candidate_name, (predictor, family) in predictors.items():
        candidate_x = predictor(z_final)
        candidate_idx, candidate_margin = top_and_margin(core, r4_bundle, candidate_x)
        candidate_pred = data.classes[candidate_idx]
        rules = {
            "ungated": np.ones(final.size, dtype=bool),
            "top_label_only": candidate_idx == global_idx,
            "top_label_plus_margin0": (candidate_idx == global_idx) & (candidate_margin >= global_margin),
        }
        for rule, accepted in rules.items():
            summary, arrays = paired_summary(core, labels_final, data.classes, global_x, candidate_x, global_pred, candidate_pred, accepted, y_final)
            all_rows.append({"candidate": candidate_name, "family": family, "commitment_rule": rule, **summary})
            detailed[(candidate_name, rule)] = arrays

    # Oracle remains a true-label diagnostic only.
    label_idx = core.class_index(labels_final, data.classes)
    oracle_x = predict_hard_residual(core, z_final, global_x, oracle_models, label_idx)
    oracle_idx, _ = top_and_margin(core, r4_bundle, oracle_x)
    oracle_summary, _ = paired_summary(core, labels_final, data.classes, global_x, oracle_x, global_pred, data.classes[oracle_idx], np.ones(final.size, dtype=bool), y_final)
    all_rows.append({"candidate": "label_conditioned_oracle_diagnostic", "family": "diagnostic", "commitment_rule": "not_deployable", **oracle_summary})

    with (args.outdir / "E1_MATCHED_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)

    # E3: validation-derived margin thresholds for the original R4 soft candidate.
    r4_global_val, r4_sem_val, r4_global_idx_val, r4_margin_global_val, r4_margin_sem_val = core.transports(r4_bundle, z_val)
    r4_sem_idx_val, _ = top_and_margin(core, r4_bundle, r4_sem_val)
    validation_diff = r4_margin_sem_val - r4_margin_global_val
    valid_diff = validation_diff[(r4_sem_idx_val == r4_global_idx_val) & (validation_diff >= 0.0)]
    thresholds = np.unique(np.concatenate(([0.0], np.quantile(valid_diff, np.linspace(0.0, 1.0, CURVE_QUANTILES)))))
    r4_global_final, r4_sem_final, r4_global_idx_final, r4_global_margin_final, r4_sem_margin_final = core.transports(r4_bundle, z_final)
    r4_sem_idx_final, _ = top_and_margin(core, r4_bundle, r4_sem_final)
    if not np.allclose(r4_global_final, global_x):
        raise RuntimeError("Selected diagonal comparator differs from R4 global map")
    r4_sem_pred_final = data.classes[r4_sem_idx_final]
    compatible = r4_sem_idx_final == r4_global_idx_final
    curve_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = compatible & ((r4_sem_margin_final - r4_global_margin_final) >= threshold)
        summary, _arrays = paired_summary(core, labels_final, data.classes, global_x, r4_sem_final, global_pred, r4_sem_pred_final, accepted, y_final)
        random_accept = stable_subset(data.loc_id[final], compatible, int(accepted.sum()))
        random_summary, _random_arrays = paired_summary(core, labels_final, data.classes, global_x, r4_sem_final, global_pred, r4_sem_pred_final, random_accept, y_final)
        curve_rows.append({"threshold": float(threshold), "selector": "margin_threshold", **summary})
        curve_rows.append({"threshold": float(threshold), "selector": "matched_coverage_hash_reference", **random_summary})
    with (args.outdir / "E3_MARGIN_RISK_COVERAGE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader(); writer.writerows(curve_rows)

    # E2 uses top-label-only and R4 margin-zero branches, both already evaluated above.
    smst_arrays = detailed[("r4_soft_decoder_weighted", "top_label_plus_margin0")]
    label_arrays = detailed[("r4_soft_decoder_weighted", "top_label_only")]
    bootstrap, bootstrap_rows = bootstrap_intervals(labels_final, data.classes, cell_ids(data.lon[final], data.lat[final]), smst_arrays, label_arrays)
    dump(args.outdir / "E2_SPATIAL_BOOTSTRAP_SUMMARY.json", bootstrap)
    with (args.outdir / "E2_SPATIAL_BOOTSTRAP_REPLICATES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader(); writer.writerows(bootstrap_rows)

    lock["final_evaluation_opened"] = True
    dump(args.outdir / "DECISION_LOCK.json", lock)
    summary = {
        "schema": "smst-review-e1e3-results-v1",
        "claim_boundary": "Post-review secondary matched analysis on the R4 paired 2021 release. It supports method attribution on this split but not new independent confirmation or population-level generalization.",
        "r4_runner_sha256": sha256(args.r4_runner),
        "r4_preregistration_sha256": sha256(args.r4_prereg),
        "decision_lock_sha256": sha256(args.outdir / "DECISION_LOCK.json"),
        "final_rows": int(final.size),
        "comparators": sorted({row["candidate"] for row in all_rows}),
        "curve_points": int(len(thresholds)),
        "spatial_bootstrap": bootstrap,
    }
    dump(args.outdir / "SUMMARY.json", summary)
    hashes = {path.name: sha256(path) for path in sorted(args.outdir.iterdir()) if path.is_file()}
    (args.outdir / "SHA256SUMS").write_text("\n".join(f"{digest}  {name}" for name, digest in hashes.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
