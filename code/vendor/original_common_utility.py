#!/usr/bin/env python3
"""Reproduce and expand the VISTA TGRS S19 common-utility experiment.

The implementation follows the manuscript's signed hash-cell split, inclusive
0.02-degree buffer, two-stage lexicographic selection, scikit-learn 1.9.0
constructors, and paired fidelity/event definitions.  It first checks the
published grid and then evaluates the reviewer-requested full-ridge and PDS
neighbors.  All held-out rows are scored only after their fold decisions are
fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score


CLASSES = np.array(["alfalfa", "almonds", "corn", "grapes", "pistachios"])
OUTER_PREFIX = "smst-r10-outer-20260815:"
INNER_PREFIX = "smst-r10-inner-20260815:"
B = 59

PUBLISHED_LOCKS = [
    {
        "stage1": {"global_alpha": 1e-6, "decoder_alpha": 1e-1},
        "full_ridge": {"alpha": 1e-2},
        "pds": {"ridge_alpha": 1e-2, "window": 1},
        "soft_residual": {"residual_alpha": 1.0},
        "logistic_full": {"residual_alpha": 1e-6, "scale": 1.0},
        "logistic_partial": {"residual_alpha": 1e-2, "scale": 0.75},
        "rf_full": {"residual_alpha": 1e-6, "scale": 1.0},
        "rf_partial": {"residual_alpha": 1e-6, "scale": 0.75},
    },
    {
        "stage1": {"global_alpha": 1e-2, "decoder_alpha": 1e-1},
        "full_ridge": {"alpha": 1.0},
        "pds": {"ridge_alpha": 1e-2, "window": 11},
        "soft_residual": {"residual_alpha": 1e-2},
        "logistic_full": {"residual_alpha": 1e-4, "scale": 1.0},
        "logistic_partial": {"residual_alpha": 1.0, "scale": 0.75},
        "rf_full": {"residual_alpha": 1.0, "scale": 1.0},
        "rf_partial": {"residual_alpha": 1e-2, "scale": 0.75},
    },
    {
        "stage1": {"global_alpha": 1e-6, "decoder_alpha": 1e-1},
        "full_ridge": {"alpha": 100.0},
        "pds": {"ridge_alpha": 1e-2, "window": 15},
        "soft_residual": {"residual_alpha": 1.0},
        "logistic_full": {"residual_alpha": 1.0, "scale": 1.0},
        "logistic_partial": {"residual_alpha": 1e-6, "scale": 0.75},
        "rf_full": {"residual_alpha": 1e-2, "scale": 1.0},
        "rf_partial": {"residual_alpha": 1e-6, "scale": 0.75},
    },
    {
        "stage1": {"global_alpha": 1e-4, "decoder_alpha": 1e-1},
        "full_ridge": {"alpha": 1.0},
        "pds": {"ridge_alpha": 1e-2, "window": 1},
        "soft_residual": {"residual_alpha": 1.0},
        "logistic_full": {"residual_alpha": 1.0, "scale": 1.0},
        "logistic_partial": {"residual_alpha": 1e-6, "scale": 0.25},
        "rf_full": {"residual_alpha": 1e-6, "scale": 1.0},
        "rf_partial": {"residual_alpha": 1.0, "scale": 0.75},
    },
    {
        "stage1": {"global_alpha": 1e-6, "decoder_alpha": 1e-1},
        "full_ridge": {"alpha": 1e-10},
        "pds": {"ridge_alpha": 1e-2, "window": 3},
        "soft_residual": {"residual_alpha": 1.0},
        "logistic_full": {"residual_alpha": 1e-6, "scale": 1.0},
        "logistic_partial": {"residual_alpha": 1e-6, "scale": 0.75},
        "rf_full": {"residual_alpha": 1e-6, "scale": 1.0},
        "rf_partial": {"residual_alpha": 1e-2, "scale": 0.75},
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def hash_mod(text: str, modulus: int = 5) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % modulus


def signed_cells(lon: np.ndarray, lat: np.ndarray, width: float) -> np.ndarray:
    ix = np.floor(lon / width).astype(np.int64)
    iy = np.floor(lat / width).astype(np.int64)
    return np.char.add(np.char.add(ix.astype(str), ":"), iy.astype(str))


def interpolate_prisma(prisma: pd.DataFrame, desis_waves: np.ndarray) -> np.ndarray:
    pcols = [c for c in prisma.columns if c.startswith("p") and c[1:].isdigit()]
    pwaves = np.array([int(c[1:]) for c in pcols], dtype=float)
    raw = prisma[pcols].to_numpy(dtype=np.float64)
    hi = np.searchsorted(pwaves, desis_waves, side="left")
    hi = np.clip(hi, 1, len(pwaves) - 1)
    lo = hi - 1
    exact = pwaves[hi] == desis_waves
    lo[exact] = hi[exact]
    denom = pwaves[hi] - pwaves[lo]
    frac = np.zeros_like(desis_waves, dtype=np.float64)
    nz = denom != 0
    frac[nz] = (desis_waves[nz] - pwaves[lo[nz]]) / denom[nz]
    return raw[:, lo] * (1.0 - frac) + raw[:, hi] * frac


def load_release(desis_path: Path, prisma_path: Path) -> dict[str, Any]:
    d = pd.read_csv(desis_path)
    p = pd.read_csv(prisma_path)
    keys = ["loc_id", "crop_name", "crop_number", "longitude", "latitude"]
    if len(d) != 16581 or len(p) != 16581:
        raise RuntimeError(f"Expected 16,581 aligned rows; got DESIS={len(d)}, PRISMA={len(p)}")
    for key in keys:
        if key in ("longitude", "latitude"):
            ok = np.array_equal(d[key].to_numpy(), p[key].to_numpy())
        else:
            ok = d[key].equals(p[key])
        if not ok:
            raise RuntimeError(f"Row-alignment check failed for {key}")
    dcols = [c for c in d.columns if c.startswith("d") and c[1:].isdigit()]
    dcols = [c for c in dcols if c != "d403"]
    dwaves = np.array([int(c[1:]) for c in dcols], dtype=float)
    if len(dcols) != B or dwaves[0] != 410 or dwaves[-1] != 1000:
        raise RuntimeError("Unexpected 59-band DESIS common grid")
    x = interpolate_prisma(p, dwaves)
    y = d[dcols].to_numpy(dtype=np.float64)
    crop = d["crop_name"].astype(str).to_numpy()
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    if set(crop) != set(CLASSES):
        raise RuntimeError(f"Unexpected crop inventory: {sorted(set(crop))}")
    y_idx = np.array([class_to_idx[v] for v in crop], dtype=np.int64)
    return {
        "loc_id": d["loc_id"].to_numpy(dtype=np.int64),
        "crop": crop,
        "class_idx": y_idx,
        "lon": d["longitude"].to_numpy(dtype=np.float64),
        "lat": d["latitude"].to_numpy(dtype=np.float64),
        "waves": dwaves,
        "x": x,
        "y": y,
    }


def fit_diag_affine(x: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None):
    n = len(x)
    if weights is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    xbar = np.sum(w[:, None] * x, axis=0)
    ybar = np.sum(w[:, None] * y, axis=0)
    xc = x - xbar
    yc = y - ybar
    slope = np.sum(w[:, None] * xc * yc, axis=0) / (
        np.sum(w[:, None] * xc * xc, axis=0) + float(alpha)
    )
    intercept = ybar - slope * xbar
    return slope, intercept


def apply_diag(x: np.ndarray, state: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    slope, intercept = state
    return x * slope + intercept


@dataclass
class Decoder:
    model: Any
    mean: np.ndarray | None
    scale: np.ndarray | None
    kind: str

    def _input(self, spectra: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return spectra
        return (spectra - self.mean) / self.scale

    def proba(self, spectra: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._input(spectra))

    def predict_idx(self, spectra: np.ndarray) -> np.ndarray:
        return np.argmax(self.proba(spectra), axis=1).astype(np.int64)


def fit_logistic(y_dev: np.ndarray, labels_dev: np.ndarray, alpha: float) -> Decoder:
    mean = y_dev.mean(axis=0)
    scale = y_dev.std(axis=0, ddof=0)
    scale = scale.copy()
    scale[scale == 0] = 1.0
    model = LogisticRegression(
        C=1.0 / float(alpha),
        class_weight="balanced",
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=0.0,
        max_iter=2000,
        n_jobs=None,
        penalty="deprecated",
        random_state=20260812,
        solver="lbfgs",
        tol=1e-4,
        verbose=0,
        warm_start=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit((y_dev - mean) / scale, CLASSES[labels_dev])
    if not np.array_equal(model.classes_, CLASSES):
        raise RuntimeError(f"Logistic class order mismatch: {model.classes_}")
    return Decoder(model=model, mean=mean, scale=scale, kind="logistic")


def fit_rf(y_dev: np.ndarray, labels_dev: np.ndarray) -> Decoder:
    model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=20260821,
        verbose=0,
        warm_start=False,
        class_weight="balanced_subsample",
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    )
    model.fit(y_dev, CLASSES[labels_dev])
    if not np.array_equal(model.classes_, CLASSES):
        raise RuntimeError(f"RF class order mismatch: {model.classes_}")
    return Decoder(model=model, mean=None, scale=None, kind="rf")


def fit_weighted_residual(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    global_dev: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    residual = y_dev - global_dev
    return [fit_diag_affine(x_dev, residual, alpha, probabilities[:, c]) for c in range(len(CLASSES))]


def apply_weighted_residual(
    x: np.ndarray,
    global_out: np.ndarray,
    probabilities: np.ndarray,
    states: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    correction = np.zeros_like(global_out)
    for c, state in enumerate(states):
        correction += probabilities[:, [c]] * apply_diag(x, state)
    return global_out + correction


def per_row_rmse(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((out - ref) ** 2, axis=1))


def per_row_sam(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(out, axis=1) * np.linalg.norm(ref, axis=1)
    cos = np.sum(out * ref, axis=1) / np.maximum(denom, 1e-15)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def evaluate(
    out: np.ndarray,
    ref: np.ndarray,
    labels: np.ndarray,
    decoder: Decoder,
    global_out: np.ndarray,
    global_pred: np.ndarray | None = None,
    commit: np.ndarray | None = None,
) -> dict[str, Any]:
    pred = decoder.predict_idx(out)
    if global_pred is None:
        global_pred = decoder.predict_idx(global_out)
    er = per_row_rmse(out, ref)
    eg = per_row_rmse(global_out, ref)
    sam = per_row_sam(out, ref)
    sg = per_row_sam(global_out, ref)
    delta = er - eg
    tail = (delta > 0.25) & (delta / np.maximum(eg, 1e-12) > 0.05)
    c2i = (global_pred == labels) & (pred != labels)
    union = tail | c2i
    if commit is None:
        commit = np.ones(len(out), dtype=bool)
    return {
        "pred": pred,
        "rmse_row": er,
        "sam_row": sam,
        "delta_rmse_row": delta,
        "delta_sam_row": sam - sg,
        "tail": tail,
        "c2i": c2i,
        "union": union,
        "label_change": pred != global_pred,
        "commit": commit,
        "f1": float(f1_score(labels, pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)),
        "rmse": float(er.mean()),
        "sam": float(sam.mean()),
        "union_count": int(union.sum()),
        "label_change_count": int(np.sum(pred != global_pred)),
        "commit_rate": float(np.mean(commit)),
    }


def selection_key(metrics: dict[str, Any], params: dict[str, Any]) -> tuple[Any, ...]:
    return (-metrics["f1"], metrics["rmse"], metrics["union_count"], metrics["sam"], canonical(params))


def fit_full_ridge(x_dev: np.ndarray, y_dev: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(
        alpha=float(alpha), fit_intercept=True, copy_X=True, max_iter=None,
        tol=1e-4, solver="auto", positive=False, random_state=None,
    )
    model.fit(x_dev, y_dev)
    return model


def fit_pds(x_dev: np.ndarray, y_dev: np.ndarray, window: int, alpha: float = 1e-2):
    states: list[tuple[int, int, Ridge]] = []
    half = int(window) // 2
    for j in range(B):
        lo = max(0, j - half)
        hi = min(B, j + half + 1)
        model = Ridge(
            alpha=float(alpha), fit_intercept=True, copy_X=True, max_iter=None,
            tol=1e-4, solver="auto", positive=False, random_state=None,
        )
        model.fit(x_dev[:, lo:hi], y_dev[:, j])
        states.append((lo, hi, model))
    return states


def apply_pds(x: np.ndarray, states) -> np.ndarray:
    out = np.empty((len(x), B), dtype=np.float64)
    for j, (lo, hi, model) in enumerate(states):
        out[:, j] = model.predict(x[:, lo:hi])
    return out


def gate_output(candidate: np.ndarray, global_out: np.ndarray, decoder: Decoder):
    global_pred = decoder.predict_idx(global_out)
    candidate_pred = decoder.predict_idx(candidate)
    commit = candidate_pred == global_pred
    out = np.where(commit[:, None], candidate, global_out)
    return out, commit, global_pred


def make_partitions(lon: np.ndarray, lat: np.ndarray) -> list[dict[str, Any]]:
    coords = np.column_stack([lon, lat])
    outer_cells = signed_cells(lon, lat, 0.05)
    outer_fold = np.array([hash_mod(OUTER_PREFIX + c) for c in outer_cells], dtype=np.int64)
    inner_cells = signed_cells(lon, lat, 0.01)
    inner_mod = np.array([hash_mod(INNER_PREFIX + c) for c in inner_cells], dtype=np.int64)
    parts = []
    for fold in range(5):
        test = outer_fold == fold
        non_test = ~test
        tree = cKDTree(coords[test])
        dist, _ = tree.query(coords[non_test], k=1, workers=-1)
        keep_non = np.zeros(len(lon), dtype=bool)
        keep_non[np.where(non_test)[0]] = dist > 0.02
        val = keep_non & (inner_mod == 0)
        dev = keep_non & (inner_mod != 0)
        parts.append({
            "fold": fold,
            "test": np.where(test)[0],
            "val": np.where(val)[0],
            "dev": np.where(dev)[0],
            "removed": int(np.sum(non_test & ~keep_non)),
            "min_retained_distance": float(np.min(dist[dist > 0.02])),
        })
    return parts


def stripped_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ["f1", "rmse", "sam", "union_count", "label_change_count", "commit_rate"]
    return {k: metrics[k] for k in keys}


def run_fold(data: dict[str, Any], part: dict[str, Any], expanded: bool,
             lock_published: bool = False) -> dict[str, Any]:
    fold = part["fold"]
    dev, val, test = part["dev"], part["val"], part["test"]
    x, y, labels = data["x"], data["y"], data["class_idx"]

    stage1_candidates = []
    for global_alpha in [1e-6, 1e-4, 1e-2, 1.0]:
        global_state = fit_diag_affine(x[dev], y[dev], global_alpha)
        g_dev = apply_diag(x[dev], global_state)
        g_val = apply_diag(x[val], global_state)
        for decoder_alpha in [1e-5, 1e-3, 1e-1]:
            decoder = fit_logistic(y[dev], labels[dev], decoder_alpha)
            metrics = evaluate(g_val, y[val], labels[val], decoder, g_val)
            params = {"global_alpha": global_alpha, "decoder_alpha": decoder_alpha}
            stage1_candidates.append((selection_key(metrics, params), params, global_state, decoder, metrics))
    stage1_candidates.sort(key=lambda z: z[0])
    stage1_selected = stage1_candidates[0]
    if lock_published:
        stage1_selected = next(z for z in stage1_candidates if z[1] == PUBLISHED_LOCKS[fold]["stage1"])
    _, base_params, global_state, logistic, base_val_metrics = stage1_selected
    rf = fit_rf(y[dev], labels[dev])

    g_dev = apply_diag(x[dev], global_state)
    g_val = apply_diag(x[val], global_state)
    g_test = apply_diag(x[test], global_state)
    log_gpred_val = logistic.predict_idx(g_val)
    rf_gpred_val = rf.predict_idx(g_val)

    family_records: dict[str, Any] = {}
    fitted: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    ridge_grid = [1e-4, 1e-2, 1.0, 100.0]
    pds_grid = [3, 7, 11]
    if expanded:
        ridge_grid = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e4, 1e6]
        pds_grid = [1, 3, 7, 11, 15, 21, 31, 41]

    candidates = []
    for alpha in ridge_grid:
        model = fit_full_ridge(x[dev], y[dev], alpha)
        out = model.predict(x[val])
        metrics = evaluate(out, y[val], labels[val], logistic, g_val, log_gpred_val)
        params = {"alpha": alpha}
        candidates.append((selection_key(metrics, params), params, model, metrics))
    candidates.sort(key=lambda z: z[0])
    selected = candidates[0]
    if lock_published:
        selected = next(z for z in candidates if z[1] == PUBLISHED_LOCKS[fold]["full_ridge"])
    _, params, model, metrics = selected
    family_records["full_ridge"] = {"params": params, "validation": stripped_metrics(metrics),
                                     "validation_grid": [{"params": p, **stripped_metrics(m)} for _, p, _, m in candidates]}
    fitted["full_ridge"] = lambda xx, m=model: m.predict(xx)

    candidates = []
    for window in pds_grid:
        state = fit_pds(x[dev], y[dev], window, 1e-2)
        out = apply_pds(x[val], state)
        metrics = evaluate(out, y[val], labels[val], logistic, g_val, log_gpred_val)
        params = {"ridge_alpha": 1e-2, "window": int(window)}
        candidates.append((selection_key(metrics, params), params, state, metrics))
    candidates.sort(key=lambda z: z[0])
    selected = candidates[0]
    if lock_published:
        selected = next(z for z in candidates if z[1] == PUBLISHED_LOCKS[fold]["pds"])
    _, params, state, metrics = selected
    family_records["pds"] = {"params": params, "validation": stripped_metrics(metrics),
                              "validation_grid": [{"params": p, **stripped_metrics(m)} for _, p, _, m in candidates]}
    fitted["pds"] = lambda xx, s=state: apply_pds(xx, s)

    def select_gated_conventional(base_name: str) -> dict[str, np.ndarray]:
        """Select conventional candidates through the same returned-output gate."""
        variants = {
            f"gated_{base_name}_full": [1.0],
            f"gated_{base_name}_partial": [0.10, 0.25, 0.50, 0.75],
        }
        candidate_lists: dict[str, list[tuple[Any, ...]]] = {name: [] for name in variants}

        if base_name == "ridge":
            raw_states = []
            for alpha in ridge_grid:
                state = fit_full_ridge(x[dev], y[dev], alpha)
                raw_states.append(({"alpha": alpha}, state, state.predict(x[val])))
        elif base_name == "pds":
            raw_states = []
            for window in pds_grid:
                state = fit_pds(x[dev], y[dev], window, 1e-2)
                raw_states.append(
                    ({"ridge_alpha": 1e-2, "window": int(window)}, state, apply_pds(x[val], state))
                )
        else:
            raise ValueError(base_name)

        for base_params, state, raw_val in raw_states:
            for name, scales in variants.items():
                for scale in scales:
                    candidate = g_val + float(scale) * (raw_val - g_val)
                    out, commit, _ = gate_output(candidate, g_val, logistic)
                    metrics = evaluate(
                        out, y[val], labels[val], logistic, g_val, log_gpred_val, commit
                    )
                    params = {**base_params, "scale": float(scale)}
                    candidate_lists[name].append(
                        (selection_key(metrics, params), params, state, metrics)
                    )

        selected_outputs: dict[str, np.ndarray] = {}
        for name, candidates_for_name in candidate_lists.items():
            candidates_for_name.sort(key=lambda z: z[0])
            _, params, state, metrics = candidates_for_name[0]
            family_records[name] = {
                "params": params,
                "validation": stripped_metrics(metrics),
                "validation_grid": [
                    {"params": p, **stripped_metrics(m)}
                    for _, p, _, m in candidates_for_name
                ],
            }
            if base_name == "ridge":
                raw_test = state.predict(x[test])
            else:
                raw_test = apply_pds(x[test], state)
            selected_outputs[name] = g_test + float(params["scale"]) * (raw_test - g_test)
        return selected_outputs

    gated_ridge_test = select_gated_conventional("ridge")
    gated_pds_test = select_gated_conventional("pds")

    log_prob_dev = logistic.proba(g_dev)
    log_prob_val = logistic.proba(g_val)
    log_prob_test = logistic.proba(g_test)
    rf_prob_dev = rf.proba(g_dev)
    rf_prob_val = rf.proba(g_val)
    rf_prob_test = rf.proba(g_test)

    candidates = []
    for residual_alpha in [1e-2, 1.0]:
        states = fit_weighted_residual(x[dev], y[dev], g_dev, log_prob_dev, residual_alpha)
        out = apply_weighted_residual(x[val], g_val, log_prob_val, states)
        metrics = evaluate(out, y[val], labels[val], logistic, g_val, log_gpred_val)
        params = {"residual_alpha": residual_alpha}
        candidates.append((selection_key(metrics, params), params, states, metrics))
    candidates.sort(key=lambda z: z[0])
    selected = candidates[0]
    if lock_published:
        selected = next(z for z in candidates if z[1] == PUBLISHED_LOCKS[fold]["soft_residual"])
    _, params, states, metrics = selected
    family_records["soft_residual"] = {"params": params, "validation": stripped_metrics(metrics),
                                        "validation_grid": [{"params": p, **stripped_metrics(m)} for _, p, _, m in candidates]}
    fitted["soft_residual"] = lambda xx, st=states: apply_weighted_residual(
        xx, apply_diag(xx, global_state), logistic.proba(apply_diag(xx, global_state)), st
    )

    def select_gated(name: str, decoder: Decoder, prob_dev: np.ndarray, prob_val: np.ndarray,
                     prob_test: np.ndarray, residual_grid: list[float], scales: list[float],
                     global_pred_val: np.ndarray):
        candidates = []
        for residual_alpha in residual_grid:
            states = fit_weighted_residual(x[dev], y[dev], g_dev, prob_dev, residual_alpha)
            full_val = apply_weighted_residual(x[val], g_val, prob_val, states)
            for scale in scales:
                candidate = g_val + float(scale) * (full_val - g_val)
                out, commit, _ = gate_output(candidate, g_val, decoder)
                metrics = evaluate(out, y[val], labels[val], decoder, g_val, global_pred_val, commit)
                params = {"residual_alpha": residual_alpha, "scale": scale}
                candidates.append((selection_key(metrics, params), params, states, metrics))
        candidates.sort(key=lambda z: z[0])
        selected = candidates[0]
        if lock_published:
            selected = next(z for z in candidates if z[1] == PUBLISHED_LOCKS[fold][name])
        _, params, states, metrics = selected
        family_records[name] = {"params": params, "validation": stripped_metrics(metrics),
                                "validation_grid": [{"params": p, **stripped_metrics(m)} for _, p, _, m in candidates]}
        full_test = apply_weighted_residual(x[test], g_test, prob_test, states)
        candidate_test = g_test + float(params["scale"]) * (full_test - g_test)
        return candidate_test

    log_full_test = select_gated("logistic_full", logistic, log_prob_dev, log_prob_val, log_prob_test,
                                  [1e-6, 1e-4, 1e-2, 1.0], [1.0], log_gpred_val)
    log_partial_test = select_gated("logistic_partial", logistic, log_prob_dev, log_prob_val, log_prob_test,
                                     [1e-6, 1e-4, 1e-2, 1.0], [0.10, 0.25, 0.50, 0.75], log_gpred_val)
    rf_full_test = select_gated("rf_full", rf, rf_prob_dev, rf_prob_val, rf_prob_test,
                                [1e-6, 1e-4, 1e-2, 1.0], [1.0], rf_gpred_val)
    rf_partial_test = select_gated("rf_partial", rf, rf_prob_dev, rf_prob_val, rf_prob_test,
                                   [1e-6, 1e-4, 1e-2, 1.0], [0.10, 0.25, 0.50, 0.75], rf_gpred_val)

    outputs: dict[str, dict[str, Any]] = {}
    log_gpred_test = logistic.predict_idx(g_test)
    rf_gpred_test = rf.predict_idx(g_test)
    outputs["logistic_global"] = evaluate(g_test, y[test], labels[test], logistic, g_test, log_gpred_test,
                                           np.zeros(len(test), dtype=bool))
    outputs["rf_global"] = evaluate(g_test, y[test], labels[test], rf, g_test, rf_gpred_test,
                                     np.zeros(len(test), dtype=bool))
    for name in ["full_ridge", "pds", "soft_residual"]:
        out = fitted[name](x[test])
        outputs[name] = evaluate(out, y[test], labels[test], logistic, g_test, log_gpred_test)
    for name, candidate, decoder, gpred in [
        ("gated_ridge_full", gated_ridge_test["gated_ridge_full"], logistic, log_gpred_test),
        ("gated_ridge_partial", gated_ridge_test["gated_ridge_partial"], logistic, log_gpred_test),
        ("gated_pds_full", gated_pds_test["gated_pds_full"], logistic, log_gpred_test),
        ("gated_pds_partial", gated_pds_test["gated_pds_partial"], logistic, log_gpred_test),
        ("logistic_full", log_full_test, logistic, log_gpred_test),
        ("logistic_partial", log_partial_test, logistic, log_gpred_test),
        ("rf_full", rf_full_test, rf, rf_gpred_test),
        ("rf_partial", rf_partial_test, rf, rf_gpred_test),
    ]:
        out, commit, _ = gate_output(candidate, g_test, decoder)
        outputs[name] = evaluate(out, y[test], labels[test], decoder, g_test, gpred, commit)

    rows = []
    for pos, idx in enumerate(test):
        row = {
            "row_index": int(idx), "loc_id": int(data["loc_id"][idx]), "fold": int(fold),
            "crop": str(data["crop"][idx]), "class_idx": int(labels[idx]),
            "longitude": float(data["lon"][idx]), "latitude": float(data["lat"][idx]),
            "outer_cell": str(signed_cells(data["lon"][idx:idx+1], data["lat"][idx:idx+1], 0.05)[0]),
        }
        for name, metrics in outputs.items():
            row[f"{name}__pred"] = int(metrics["pred"][pos])
            row[f"{name}__rmse"] = float(metrics["rmse_row"][pos])
            row[f"{name}__sam"] = float(metrics["sam_row"][pos])
            row[f"{name}__delta_rmse"] = float(metrics["delta_rmse_row"][pos])
            row[f"{name}__delta_sam"] = float(metrics["delta_sam_row"][pos])
            row[f"{name}__tail"] = int(metrics["tail"][pos])
            row[f"{name}__c2i"] = int(metrics["c2i"][pos])
            row[f"{name}__union"] = int(metrics["union"][pos])
            row[f"{name}__label_change"] = int(metrics["label_change"][pos])
            row[f"{name}__commit"] = int(metrics["commit"][pos])
        rows.append(row)

    return {
        "fold": fold,
        "geometry": {
            "test": len(test), "validation": len(val), "development": len(dev),
            "removed_by_buffer": part["removed"], "min_retained_distance": part["min_retained_distance"],
            "class_counts": {
                split_name: [int(np.sum(labels[idxs] == c)) for c in range(len(CLASSES))]
                for split_name, idxs in [("development", dev), ("validation", val), ("test", test)]
            },
        },
        "stage1": {"params": base_params, "validation": stripped_metrics(base_val_metrics)},
        "families": family_records,
        "rows": rows,
    }


def outer_cell_bootstrap(df: pd.DataFrame, column: str, seed: int, reps: int = 2000) -> dict[str, Any]:
    groups = {cell: g[column].to_numpy(dtype=np.float64) for cell, g in df.groupby("outer_cell", sort=True)}
    cells = list(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        sampled = rng.choice(cells, size=len(cells), replace=True)
        total = 0.0
        count = 0
        for cell in sampled:
            arr = groups[cell]
            total += float(arr.sum())
            count += len(arr)
        values[r] = total / count
    return {
        "observed": float(df[column].mean()),
        "bootstrap_mean": float(values.mean()),
        "ci95": [float(v) for v in np.percentile(values, [2.5, 97.5])],
        "units": len(cells), "reps": reps, "seed": seed,
    }


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    results = {}
    for name in [
        "logistic_global", "full_ridge", "pds", "soft_residual",
        "gated_ridge_full", "gated_ridge_partial", "gated_pds_full", "gated_pds_partial",
        "logistic_full", "logistic_partial", "rf_global", "rf_full", "rf_partial",
    ]:
        pred = df[f"{name}__pred"].to_numpy(dtype=int)
        labels = df["class_idx"].to_numpy(dtype=int)
        results[name] = {
            "commit_rate": float(df[f"{name}__commit"].mean()),
            "macro_f1": float(f1_score(labels, pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)),
            "mean_rmse": float(df[f"{name}__rmse"].mean()),
            "mean_sam": float(df[f"{name}__sam"].mean()),
            "delta_rmse": float(df[f"{name}__delta_rmse"].mean()),
            "delta_sam": float(df[f"{name}__delta_sam"].mean()),
            "label_changes": int(df[f"{name}__label_change"].sum()),
            "tail_events": int(df[f"{name}__tail"].sum()),
            "correct_to_incorrect": int(df[f"{name}__c2i"].sum()),
            "union_events": int(df[f"{name}__union"].sum()),
            "outer_cell_delta_rmse": outer_cell_bootstrap(df, f"{name}__delta_rmse", 20260816),
            "outer_cell_delta_sam": outer_cell_bootstrap(df, f"{name}__delta_sam", 20260817),
        }
    df = df.copy()
    df["log_partial_minus_full_rmse"] = df["logistic_partial__rmse"] - df["logistic_full__rmse"]
    df["log_partial_minus_full_sam"] = df["logistic_partial__sam"] - df["logistic_full__sam"]
    df["log_partial_minus_full_commit"] = df["logistic_partial__commit"] - df["logistic_full__commit"]
    df["log_partial_minus_full_union_rate"] = df["logistic_partial__union"] - df["logistic_full__union"]
    df["rf_partial_minus_full_rmse"] = df["rf_partial__rmse"] - df["rf_full__rmse"]
    df["rf_partial_minus_full_sam"] = df["rf_partial__sam"] - df["rf_full__sam"]
    df["rf_partial_minus_full_commit"] = df["rf_partial__commit"] - df["rf_full__commit"]
    df["rf_partial_minus_full_union_rate"] = df["rf_partial__union"] - df["rf_full__union"]
    paired = {}
    for key in ["log_partial_minus_full_rmse", "log_partial_minus_full_sam", "log_partial_minus_full_commit",
                "log_partial_minus_full_union_rate", "rf_partial_minus_full_rmse", "rf_partial_minus_full_sam",
                "rf_partial_minus_full_commit", "rf_partial_minus_full_union_rate"]:
        seed = 20260817 if key.endswith("sam") else 20260821
        paired[key] = outer_cell_bootstrap(df, key, seed)
    paired["log_partial_minus_full_union_count"] = int(
        df["logistic_partial__union"].sum() - df["logistic_full__union"].sum())
    paired["rf_partial_minus_full_union_count"] = int(
        df["rf_partial__union"].sum() - df["rf_full__union"].sum())

    control_pairs = {
        "ridge_partial_minus_full": ("gated_ridge_partial", "gated_ridge_full"),
        "pds_partial_minus_full": ("gated_pds_partial", "gated_pds_full"),
        "smst_partial_minus_ridge_partial": ("logistic_partial", "gated_ridge_partial"),
        "smst_partial_minus_pds_partial": ("logistic_partial", "gated_pds_partial"),
    }
    for label, (left, right) in control_pairs.items():
        for metric, suffix in [
            ("rmse", "rmse"),
            ("sam", "sam"),
            ("commit", "commit"),
            ("union", "union_rate"),
        ]:
            key = f"{label}_{suffix}"
            df[key] = df[f"{left}__{metric}"] - df[f"{right}__{metric}"]
            paired[key] = outer_cell_bootstrap(df, key, 20260922)
        paired[f"{label}_union_count"] = int(
            df[f"{left}__union"].sum() - df[f"{right}__union"].sum()
        )
    return {"results": results, "paired": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desis", type=Path, required=True)
    parser.add_argument("--prisma", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--lock-published", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if sklearn.__version__ != "1.9.0":
        raise RuntimeError(f"This record requires scikit-learn 1.9.0, found {sklearn.__version__}")
    data = load_release(args.desis, args.prisma)
    partitions = make_partitions(data["lon"], data["lat"])

    fold_records = []
    row_frames = []
    for part in partitions:
        print(f"[fold {part['fold']}] start", flush=True)
        record = run_fold(data, part, args.expanded, args.lock_published)
        rows = pd.DataFrame(record.pop("rows"))
        row_frames.append(rows)
        fold_records.append(record)
        (args.out / f"fold_{part['fold']}_decision.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        rows.to_csv(args.out / f"fold_{part['fold']}_heldout.csv", index=False)
        print(f"[fold {part['fold']}] done", flush=True)

    heldout = pd.concat(row_frames, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    if len(heldout) != 16581 or heldout["row_index"].nunique() != 16581:
        raise RuntimeError("Held-out concatenation does not contain each release row exactly once")
    heldout.to_csv(args.out / "heldout_per_location.csv", index=False)
    summary = summarize(heldout)
    summary["expanded"] = bool(args.expanded)
    summary["lock_published"] = bool(args.lock_published)
    summary["runtime"] = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    }
    summary["inputs"] = {
        "desis": str(args.desis.resolve()), "desis_sha256": sha256(args.desis),
        "prisma": str(args.prisma.resolve()), "prisma_sha256": sha256(args.prisma),
    }
    summary["fold_decisions"] = fold_records
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "expanded": args.expanded,
        "lock_published": args.lock_published,
        "stage1": [r["stage1"]["params"] for r in fold_records],
        "families": [{k: v["params"] for k, v in r["families"].items()} for r in fold_records],
        "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("outer_cell")}
                    for k, v in summary["results"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
