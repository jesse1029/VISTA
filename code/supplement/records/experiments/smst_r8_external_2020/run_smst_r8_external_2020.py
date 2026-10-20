#!/usr/bin/env python3
"""R8: frozen-2021 comparisons on the external 2020 DESIS--PRISMA release.

All baseline selection uses only the already defined R4 development and
validation partitions.  The 2021 final partition is intentionally never read.
The 2020 release is evaluated once after these choices are fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score


SEED = 20260812
EXTERNAL_HASHES = {
    "desis_2020.csv": "2c9a126a77baada7d74d6866b204775a18b0ccfb515c42611f33a02cece6907c",
    "prisma_2020.csv": "3ed9d66f082dffba937ac1a4f743d651432900fa26ca582fd7c0dc963a44071b",
    "metadata_2020.xml": "89fbc0d64c2137e9eaaebfd647436a82146a5c677ca8852f2818c564ae341f22",
}
RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1.0)
PLS_COMPONENTS = (2, 4, 8, 16, 32)
PDS_HALF_WIDTHS = (1, 3, 5)
CLUSTER_COUNTS = (2, 3, 5, 7)
TEMPERATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
BOOTSTRAPS = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def band_columns(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    columns = [c for c in frame.columns if c.startswith("X") and c[1:].isdigit()]
    wavelengths = np.asarray([float(c[1:]) for c in columns], dtype=np.float64)
    if len(columns) < 50 or not np.all(np.diff(wavelengths) > 0):
        raise RuntimeError("External spectral columns are absent or not strictly ordered")
    return columns, wavelengths


def interpolate_rows(values: np.ndarray, source_wave: np.ndarray, target_wave: np.ndarray) -> np.ndarray:
    if target_wave.min() < source_wave.min() or target_wave.max() > source_wave.max():
        raise RuntimeError(
            f"Fixed R4 wavelengths [{target_wave.min()}, {target_wave.max()}] "
            f"fall outside external source support [{source_wave.min()}, {source_wave.max()}]"
        )
    return np.vstack([np.interp(target_wave, source_wave, row) for row in values])


@dataclass(frozen=True)
class ExternalData:
    loc_id: np.ndarray
    crop: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    prisma: np.ndarray
    desis: np.ndarray
    wavelength_nm: np.ndarray


def load_external(root: Path, target_wave: np.ndarray) -> ExternalData:
    actual = {name: sha256(root / name) for name in EXTERNAL_HASHES}
    if actual != EXTERNAL_HASHES:
        raise RuntimeError(f"External input hash mismatch: {actual}")
    desis = pd.read_csv(root / "desis_2020.csv")
    prisma = pd.read_csv(root / "prisma_2020.csv")
    required = {"Unique ID", "Longitude", "Latitude", "Image ID", "Crop Type"}
    if not required.issubset(desis.columns) or not required.issubset(prisma.columns):
        raise RuntimeError("External metadata columns are incomplete")
    desis = desis.loc[desis["Image ID"].astype(str).str.contains("20200618", regex=False)].copy()
    prisma = prisma.loc[prisma["Image ID"].astype(str).str.contains("20200617", regex=False)].copy()
    if len(desis) != 1490 or len(prisma) != 1490:
        raise RuntimeError(f"Expected the June paired subset, got DESIS={len(desis)} PRISMA={len(prisma)}")
    desis = desis.sort_values("Unique ID").reset_index(drop=True)
    prisma = prisma.sort_values("Unique ID").reset_index(drop=True)
    for name in ("Unique ID", "Longitude", "Latitude", "Crop Type"):
        if not np.array_equal(desis[name].to_numpy(), prisma[name].to_numpy()):
            raise RuntimeError(f"External pairing mismatch in {name}")
    dcols, dwave = band_columns(desis)
    pcols, pwave = band_columns(prisma)
    return ExternalData(
        loc_id=desis["Unique ID"].to_numpy(dtype=np.int64),
        crop=desis["Crop Type"].astype(str).to_numpy(),
        lon=desis["Longitude"].to_numpy(dtype=np.float64),
        lat=desis["Latitude"].to_numpy(dtype=np.float64),
        prisma=interpolate_rows(prisma[pcols].to_numpy(dtype=np.float64), pwave, target_wave),
        desis=interpolate_rows(desis[dcols].to_numpy(dtype=np.float64), dwave, target_wave),
        wavelength_nm=target_wave.copy(),
    )


def rmse_per_location(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((x - y) ** 2, axis=1))


def sam_per_location(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    cosine = np.divide(np.sum(x * y, axis=1), denom, out=np.ones_like(denom), where=denom > 0)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def cell_ids(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{int(np.floor((x + 180.0) / .01))}:{int(np.floor((y + 90.0) / .01))}" for x, y in zip(lon, lat, strict=True)],
        dtype=str,
    )


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    adjusted = scores / float(temperature)
    adjusted = adjusted - adjusted.max(axis=1, keepdims=True)
    exps = np.exp(adjusted)
    return exps / exps.sum(axis=1, keepdims=True)


def margin_and_top(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, axis=1, kind="mergesort")
    top = order[:, -1]
    margin = scores[np.arange(scores.shape[0]), top] - scores[np.arange(scores.shape[0]), order[:, -2]]
    return top, margin


def select_lowest(rows: list[dict[str, Any]], config_keys: tuple[str, ...]) -> dict[str, Any]:
    return min(rows, key=lambda r: tuple([float(r["val_rmse"])] + [r[k] for k in config_keys]))


def fit_diagonal(core, x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    a, b = core.fit_diagonal_affine(x, y, alpha)
    return {"kind": "diagonal", "alpha": alpha, "a": a, "b": b}


def predict_model(core, model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    kind = model["kind"]
    if kind == "diagonal":
        return core.apply_affine(x, model["a"], model["b"])
    if kind == "full_ridge":
        return model["model"].predict(x)
    if kind == "pls":
        return model["model"].predict(x)
    if kind == "pds":
        out = np.empty((x.shape[0], len(model["models"])), dtype=np.float64)
        width = int(model["half_width"])
        for j, reg in enumerate(model["models"]):
            lo, hi = max(0, j - width), min(x.shape[1], j + width + 1)
            out[:, j] = reg.predict(x[:, lo:hi])
        return out
    if kind == "unconditional_residual":
        base = core.apply_affine(x, model["base_a"], model["base_b"])
        return base + core.apply_affine(x, model["residual_a"], model["residual_b"])
    if kind == "cluster_residual":
        base = core.apply_affine(x, model["base_a"], model["base_b"])
        assignment = model["kmeans"].predict(x)
        residual = np.empty_like(base)
        for cluster, (a, b) in enumerate(zip(model["residual_a"], model["residual_b"], strict=True)):
            rows = assignment == cluster
            if rows.any():
                residual[rows] = core.apply_affine(x[rows], a, b)
        return base + residual
    raise RuntimeError(f"Unknown model kind: {kind}")


def fit_full_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    model = Ridge(alpha=alpha, fit_intercept=True).fit(x, y)
    return {"kind": "full_ridge", "alpha": alpha, "model": model}


def fit_pls(x: np.ndarray, y: np.ndarray, components: int) -> dict[str, Any]:
    model = PLSRegression(n_components=components, scale=True, max_iter=2000, tol=1e-10).fit(x, y)
    return {"kind": "pls", "components": components, "model": model}


def fit_pds(x: np.ndarray, y: np.ndarray, half_width: int, alpha: float) -> dict[str, Any]:
    models = []
    for j in range(y.shape[1]):
        lo, hi = max(0, j - half_width), min(x.shape[1], j + half_width + 1)
        models.append(Ridge(alpha=alpha, fit_intercept=True).fit(x[:, lo:hi], y[:, j]))
    return {"kind": "pds", "half_width": half_width, "alpha": alpha, "models": models}


def fit_unconditional_residual(core, x: np.ndarray, y: np.ndarray, base: dict[str, Any], alpha: float) -> dict[str, Any]:
    baseline = core.apply_affine(x, base["a"], base["b"])
    a, b = core.fit_diagonal_affine(x, y - baseline, alpha)
    return {
        "kind": "unconditional_residual",
        "alpha": alpha,
        "base_a": base["a"],
        "base_b": base["b"],
        "residual_a": a,
        "residual_b": b,
    }


def fit_cluster_residual(core, x: np.ndarray, y: np.ndarray, base: dict[str, Any], count: int, alpha: float) -> dict[str, Any]:
    kmeans = KMeans(n_clusters=count, n_init=20, random_state=SEED).fit(x)
    baseline = core.apply_affine(x, base["a"], base["b"])
    residual_a, residual_b = [], []
    for cluster in range(count):
        rows = kmeans.labels_ == cluster
        if int(rows.sum()) < 10:
            raise RuntimeError("Cluster support below 10 in development")
        a, b = core.fit_diagonal_affine(x[rows], (y - baseline)[rows], alpha)
        residual_a.append(a)
        residual_b.append(b)
    return {
        "kind": "cluster_residual",
        "clusters": count,
        "alpha": alpha,
        "base_a": base["a"],
        "base_b": base["b"],
        "kmeans": kmeans,
        "residual_a": np.stack(residual_a),
        "residual_b": np.stack(residual_b),
    }


def parameter_count(model: dict[str, Any]) -> int:
    kind = model["kind"]
    if kind == "diagonal":
        return int(model["a"].size + model["b"].size)
    if kind == "full_ridge":
        return int(model["model"].coef_.size + model["model"].intercept_.size)
    if kind == "pls":
        return int(model["model"].coef_.size + np.asarray(model["model"].intercept_).size)
    if kind == "pds":
        return int(sum(reg.coef_.size + 1 for reg in model["models"]))
    if kind == "unconditional_residual":
        return int(model["base_a"].size + model["base_b"].size + model["residual_a"].size + model["residual_b"].size)
    if kind == "cluster_residual":
        return int(
            model["base_a"].size + model["base_b"].size + model["kmeans"].cluster_centers_.size
            + model["residual_a"].size + model["residual_b"].size
        )
    raise RuntimeError(kind)


def validation_select(core, data, dev: np.ndarray, val: np.ndarray) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    xdev, ydev = data.z_prisma[dev], data.y_desis[dev]
    xval, yval = data.z_prisma[val], data.y_desis[val]
    selected: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {"selection_partition": "R4 validation only", "rows": {}}

    diagonal_rows: list[dict[str, Any]] = []
    for alpha in RIDGE_GRID:
        model = fit_diagonal(core, xdev, ydev, alpha)
        diagonal_rows.append({"alpha": alpha, "val_rmse": float(rmse_per_location(predict_model(core, model, xval), yval).mean()), "model": model})
    choice = select_lowest(diagonal_rows, ("alpha",))
    base = choice.pop("model")
    selected["diagonal_affine"] = base
    manifest["rows"]["diagonal_affine"] = {k: v for k, v in choice.items()}

    full_rows: list[dict[str, Any]] = []
    for alpha in RIDGE_GRID:
        model = fit_full_ridge(xdev, ydev, alpha)
        full_rows.append({"alpha": alpha, "val_rmse": float(rmse_per_location(predict_model(core, model, xval), yval).mean()), "model": model})
    choice = select_lowest(full_rows, ("alpha",))
    selected["full_ridge"] = choice.pop("model")
    manifest["rows"]["full_ridge"] = {k: v for k, v in choice.items()}

    pls_rows: list[dict[str, Any]] = []
    for components in PLS_COMPONENTS:
        model = fit_pls(xdev, ydev, components)
        pls_rows.append({"components": components, "val_rmse": float(rmse_per_location(predict_model(core, model, xval), yval).mean()), "model": model})
    choice = select_lowest(pls_rows, ("components",))
    selected["pls"] = choice.pop("model")
    manifest["rows"]["pls"] = {k: v for k, v in choice.items()}

    pds_rows: list[dict[str, Any]] = []
    for width in PDS_HALF_WIDTHS:
        for alpha in RIDGE_GRID:
            model = fit_pds(xdev, ydev, width, alpha)
            pds_rows.append({"half_width": width, "alpha": alpha, "val_rmse": float(rmse_per_location(predict_model(core, model, xval), yval).mean()), "model": model})
    choice = select_lowest(pds_rows, ("half_width", "alpha"))
    selected["pds"] = choice.pop("model")
    manifest["rows"]["pds"] = {k: v for k, v in choice.items()}

    residual_rows: list[dict[str, Any]] = []
    for alpha in RIDGE_GRID:
        model = fit_unconditional_residual(core, xdev, ydev, base, alpha)
        residual_rows.append({"alpha": alpha, "val_rmse": float(rmse_per_location(predict_model(core, model, xval), yval).mean()), "model": model})
    choice = select_lowest(residual_rows, ("alpha",))
    selected["unconditional_residual"] = choice.pop("model")
    manifest["rows"]["unconditional_residual"] = {k: v for k, v in choice.items()}

    cluster_rows: list[dict[str, Any]] = []
    for clusters in CLUSTER_COUNTS:
        for alpha in RIDGE_GRID:
            model = fit_cluster_residual(core, xdev, ydev, base, clusters, alpha)
            cluster_rows.append({"clusters": clusters, "alpha": alpha, "val_rmse": float(rmse_per_location(predict_model(core, model, xval), yval).mean()), "model": model})
    choice = select_lowest(cluster_rows, ("clusters", "alpha"))
    selected["cluster_residual"] = choice.pop("model")
    manifest["rows"]["cluster_residual"] = {k: v for k, v in choice.items()}
    return selected, manifest


def deterministic_top(scores: np.ndarray, loc_id: np.ndarray, eligible: np.ndarray, count: int) -> np.ndarray:
    chosen = np.zeros(scores.shape[0], dtype=bool)
    candidates = np.flatnonzero(eligible)
    if count > candidates.size:
        raise RuntimeError(f"Cannot select {count} rows from {candidates.size} eligible candidates")
    order = np.lexsort((loc_id[candidates], -np.nan_to_num(scores[candidates], nan=-np.inf)))
    chosen[candidates[order[:count]]] = True
    return chosen


def fit_selector_lock(core, data, val: np.ndarray, bundle: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    gx, sx, global_idx, global_margin, semantic_margin = core.transports(bundle, data.z_prisma[val])
    semantic_scores = core.logits(bundle, sx)
    semantic_idx, _ = margin_and_top(semantic_scores)
    same_label = semantic_idx == global_idx
    both = same_label & (semantic_margin >= global_margin)
    count = int(both.sum())
    if count < 1:
        raise RuntimeError("R4 validation has no SMST commitments")
    labels = np.asarray([np.where(data.classes == value)[0][0] for value in data.crop[val]], dtype=np.int64)
    temp_rows = []
    for temperature in TEMPERATURE_GRID:
        probs = softmax(semantic_scores, temperature)
        nll = -np.log(np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)).mean()
        temp_rows.append((float(nll), float(temperature)))
    _, temperature = min(temp_rows)
    probability = softmax(semantic_scores, temperature).max(axis=1)
    entropy = -(softmax(semantic_scores, temperature) * np.log(np.clip(softmax(semantic_scores, temperature), 1e-12, 1.0))).sum(axis=1)
    normalized_margin = semantic_margin / np.maximum(np.linalg.norm(semantic_scores, axis=1), 1e-12)
    return {
        "temperature": temperature,
        "validation_both_count": count,
        "validation_top_label_count": int(same_label.sum()),
        "validation_conditional_acceptance_fraction": float(count / same_label.sum()),
        "validation_rows": int(val.size),
        "validation_both_coverage": float(both.mean()),
        "rules": {
            "top_label_only": {"kind": "logical", "matched": False},
            "margin_only": {"kind": "logical", "matched": False},
            "both_raw_margin": {"kind": "logical", "matched": False},
            "top_label_temperature_probability": {"kind": "rank", "direction": "descending", "matched": True},
            "top_label_entropy": {"kind": "rank", "direction": "ascending", "matched": True},
            "top_label_normalized_margin": {"kind": "rank", "direction": "descending", "matched": True},
        },
    }, both


def shared_macro_f1(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray) -> tuple[float | None, int]:
    shared = np.asarray([value in set(classes.tolist()) for value in labels], dtype=bool)
    if not shared.any():
        return None, 0
    active = np.asarray(sorted(set(labels[shared])), dtype=str)
    return float(f1_score(labels[shared], predictions[shared], labels=active, average="macro", zero_division=0)), int(shared.sum())


def bootstrap_delta(cell: np.ndarray, delta_rmse: np.ndarray, delta_sam: np.ndarray, committed: np.ndarray, harm: np.ndarray) -> dict[str, Any]:
    unique, inverse = np.unique(cell, return_inverse=True)
    block_rows = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    rng = np.random.default_rng(SEED)
    draws = np.empty((BOOTSTRAPS, 4), dtype=np.float64)
    for replicate in range(BOOTSTRAPS):
        selected = rng.integers(0, unique.size, size=unique.size)
        rows = np.concatenate([block_rows[i] for i in selected])
        draws[replicate] = [delta_rmse[rows].mean(), delta_sam[rows].mean(), committed[rows].mean(), harm[rows].mean()]
    names = ("delta_rmse", "delta_sam_rad", "coverage", "harm_rate_all")
    return {name: {"mean": float(draws[:, j].mean()), "ci95": [float(v) for v in np.quantile(draws[:, j], (0.025, 0.975))]} for j, name in enumerate(names)}


def selector_result(core, name: str, accepted: np.ndarray, gx: np.ndarray, sx: np.ndarray, global_idx: np.ndarray,
                    semantic_idx: np.ndarray, labels: np.ndarray, reference: np.ndarray, classes: np.ndarray,
                    external_cells: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    chosen = np.where(accepted[:, None], sx, gx)
    prediction_idx = np.where(accepted, semantic_idx, global_idx)
    prediction = classes[prediction_idx]
    global_rmse = rmse_per_location(gx, reference)
    global_sam = sam_per_location(gx, reference)
    rmse = rmse_per_location(chosen, reference)
    sam = sam_per_location(chosen, reference)
    harm = ((rmse - global_rmse > 0.25) & ((rmse - global_rmse) / np.maximum(global_rmse, 1e-12) > 0.05))
    shared_f1, shared_n = shared_macro_f1(labels, prediction, classes)
    payload = {
        "name": name,
        "coverage": float(accepted.mean()),
        "accepted_rows": int(accepted.sum()),
        "frozen_label_changes_vs_global": int((prediction_idx != global_idx).sum()),
        "spectral_rmse": float(rmse.mean()),
        "delta_rmse_vs_global": float(rmse.mean() - global_rmse.mean()),
        "sam_rad": float(sam.mean()),
        "delta_sam_rad_vs_global": float(sam.mean() - global_sam.mean()),
        "shared_label_macro_f1": shared_f1,
        "shared_label_rows": shared_n,
        "declared_harms_vs_global": int(harm.sum()),
        "harm_rate_all": float(harm.mean()),
        "harm_rate_committed": float(harm[accepted].mean()) if accepted.any() else None,
        "spatial_cell_bootstrap": bootstrap_delta(external_cells, rmse - global_rmse, sam - global_sam, accepted, harm),
    }
    return payload, chosen, rmse, harm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4-runner", required=True, type=Path)
    parser.add_argument("--r4-prereg", required=True, type=Path)
    parser.add_argument("--r4-development", required=True, type=Path)
    parser.add_argument("--r4-validation", required=True, type=Path)
    parser.add_argument("--r4-data", required=True, type=Path)
    parser.add_argument("--external-data", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    if args.outdir.exists():
        raise RuntimeError(f"Output already exists: {args.outdir}")
    if sha256(args.prereg) != sha256(Path(__file__).with_name("PREREGISTRATION.md")):
        raise RuntimeError("The supplied preregistration is not the runner's adjacent authority")
    sys.path.insert(0, str(args.r4_runner.parent))
    import run_smst_r4 as core  # noqa: PLC0415

    source = core.load_data(args.r4_data.resolve())
    development, validation = core.indices(source, "development"), core.indices(source, "validation")
    decision = json.loads((args.r4_validation / "FINAL_DECISION_LOCK.json").read_text(encoding="utf-8"))
    if decision["script_sha256"] != sha256(args.r4_runner) or decision["pre_registration_sha256"] != sha256(args.r4_prereg):
        raise RuntimeError("R4 decision lock does not bind the supplied authority")
    with (args.r4_development / "development_models.pkl").open("rb") as handle:
        r4_bundle = pickle.load(handle)[decision["selected"]["model_key"]]

    external = load_external(args.external_data.resolve(), source.desis_wavelength_nm)
    args.outdir.mkdir(parents=True)
    started = time.perf_counter()
    baselines, selection_manifest = validation_select(core, source, development, validation)
    selection_seconds = time.perf_counter() - started
    write_json(args.outdir / "R8_SELECTION.json", selection_manifest)

    baseline_summary: dict[str, Any] = {}
    baseline_outputs: dict[str, np.ndarray] = {}
    for name, model in baselines.items():
        clock = time.perf_counter()
        output = predict_model(core, model, external.prisma)
        elapsed = time.perf_counter() - clock
        rmse = rmse_per_location(output, external.desis)
        sam = sam_per_location(output, external.desis)
        baseline_outputs[name] = output
        baseline_summary[name] = {
            "spectral_rmse": float(rmse.mean()),
            "sam_rad": float(sam.mean()),
            "parameter_count": parameter_count(model),
            "external_evaluation_seconds": elapsed,
        }
    diagonal = baseline_outputs["diagonal_affine"]

    gx, sx, global_idx, global_margin, semantic_margin = core.transports(r4_bundle, external.prisma)
    semantic_scores = core.logits(r4_bundle, sx)
    semantic_idx, _ = margin_and_top(semantic_scores)
    same_label = semantic_idx == global_idx
    selector_lock, _ = fit_selector_lock(core, source, validation, r4_bundle)
    matched_count = int(round(selector_lock["validation_conditional_acceptance_fraction"] * same_label.sum()))
    temperature = float(selector_lock["temperature"])
    probabilities = softmax(semantic_scores, temperature)
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    norm_margin = semantic_margin / np.maximum(np.linalg.norm(semantic_scores, axis=1), 1e-12)
    external_cell = cell_ids(external.lon, external.lat)
    acceptance = {
        "always_residual": np.ones(external.loc_id.size, dtype=bool),
        "top_label_only": same_label,
        "margin_only": semantic_margin >= global_margin,
        "smst_top_label_and_raw_margin": same_label & (semantic_margin >= global_margin),
        "top_label_temperature_probability_matched": deterministic_top(confidence, external.loc_id, same_label, matched_count),
        "top_label_entropy_matched": deterministic_top(-entropy, external.loc_id, same_label, matched_count),
        "top_label_normalized_margin_matched": deterministic_top(norm_margin, external.loc_id, same_label, matched_count),
    }
    selector_summary: dict[str, Any] = {}
    per_location: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, mask in acceptance.items():
        result, output, rmse, harm = selector_result(
            core, name, mask, gx, sx, global_idx, semantic_idx, external.crop, external.desis, source.classes, external_cell
        )
        selector_summary[name] = result
        per_location[name] = (mask, rmse, harm)
    per_wavelength = pd.DataFrame({"wavelength_nm": external.wavelength_nm})
    for name, output in {"diagonal_affine": diagonal, "smst": per_location["smst_top_label_and_raw_margin"][0][:, None] * sx + (~per_location["smst_top_label_and_raw_margin"][0])[:, None] * gx}.items():
        error = output - external.desis
        per_wavelength[f"{name}_bias"] = error.mean(axis=0)
        per_wavelength[f"{name}_rmse"] = np.sqrt((error ** 2).mean(axis=0))
    per_wavelength.to_csv(args.outdir / "R8_WAVELENGTH_DIAGNOSTICS.csv", index=False)

    with (args.outdir / "R8_PER_LOCATION.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["loc_id", "crop_label", "longitude", "latitude", "spatial_cell", "global_prediction", "semantic_prediction", "global_margin", "semantic_margin", "global_rmse"]
        for name in acceptance:
            fields.extend([f"{name}_accept", f"{name}_rmse", f"{name}_harm"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        global_rmse = rmse_per_location(gx, external.desis)
        for i in range(external.loc_id.size):
            row = {
                "loc_id": int(external.loc_id[i]),
                "crop_label": external.crop[i],
                "longitude": float(external.lon[i]),
                "latitude": float(external.lat[i]),
                "spatial_cell": external_cell[i],
                "global_prediction": source.classes[global_idx[i]],
                "semantic_prediction": source.classes[semantic_idx[i]],
                "global_margin": float(global_margin[i]),
                "semantic_margin": float(semantic_margin[i]),
                "global_rmse": float(global_rmse[i]),
            }
            for name, (mask, rmse, harm) in per_location.items():
                row[f"{name}_accept"] = bool(mask[i])
                row[f"{name}_rmse"] = float(rmse[i])
                row[f"{name}_harm"] = bool(harm[i])
            writer.writerow(row)

    summary = {
        "schema": "smst-r8-external-2020-v1",
        "claim_boundary": (
            "Frozen-2021 evaluation on a separate 2020 USGS DESIS--PRISMA release. "
            "It is external temporal evidence with the same sensor pair and California agricultural setting; "
            "it is not an independent sensor-pair, population-safety, or universal-generalization certificate."
        ),
        "input_sha256": {
            "r4_runner": sha256(args.r4_runner),
            "r4_preregistration": sha256(args.r4_prereg),
            "r8_runner": sha256(Path(__file__).resolve()),
            "r8_preregistration": sha256(args.prereg),
            "r4_development_models": sha256(args.r4_development / "development_models.pkl"),
            "r4_decision_lock": sha256(args.r4_validation / "FINAL_DECISION_LOCK.json"),
            **EXTERNAL_HASHES,
        },
        "r4_selected": decision["selected"],
        "external_pair": {
            "release_doi": "10.5066/P98LO5D4",
            "desis_image": "DESIS-HSI-L2A-DT0466155472_005-20200618T152131-V0210",
            "prisma_image": "PRS_L2D_STD_20200617185821_20200617185825_0001",
            "rows": int(external.loc_id.size),
            "shared_decoder_label_rows": int(sum(value in set(source.classes.tolist()) for value in external.crop)),
            "spatial_cells": int(np.unique(external_cell).size),
            "wavelength_count": int(external.wavelength_nm.size),
        },
        "baseline_selection_seconds": selection_seconds,
        "baseline_external": baseline_summary,
        "selector_lock": selector_lock,
        "selectors_external": selector_summary,
    }
    write_json(args.outdir / "R8_EXTERNAL_SUMMARY.json", summary)
    files = {p.name: sha256(p) for p in sorted(args.outdir.iterdir()) if p.is_file()}
    (args.outdir / "SHA256SUMS").write_text("\n".join(f"{digest}  {name}" for name, digest in files.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
