#!/usr/bin/env python3
"""Deterministic paired-distribution and gate-frontier figure for R4/R6/R8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "unconstrained": "#9A4D8E",
    "label_only": "#5B8DB8",
    "margin_only": "#C37B2A",
    "both": "#0F4D92",
    "external": "#B64342",
    "ink": "#202124",
    "grid": "#D9D9D9",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def apply_variant(rows: pd.DataFrame, accept_column: str) -> np.ndarray:
    return np.where(rows[accept_column].to_numpy(bool), rows.semantic_rmse, rows.global_rmse) - rows.global_rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6-summary", type=Path, required=True)
    parser.add_argument("--r6-per-location", type=Path, required=True)
    parser.add_argument("--r8-summary", type=Path, required=True)
    parser.add_argument("--r8-per-location", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    r6_summary = json.loads(args.r6_summary.read_text(encoding="utf-8"))
    r8_summary = json.loads(args.r8_summary.read_text(encoding="utf-8"))
    r6_rows = pd.read_csv(args.r6_per_location)
    r8_rows = pd.read_csv(args.r8_per_location)

    variants = [
        ("unconstrained", "Unconstrained", "accept_unconstrained"),
        ("label_only", "Top label only", "accept_label_only"),
        ("margin_only", "Margin only", "accept_margin_only"),
        ("both", "SMST conjunction", "accept_both"),
    ]
    deltas = {key: apply_variant(r6_rows, column) for key, _, column in variants}
    r8_delta = (
        r8_rows.smst_top_label_and_raw_margin_rmse.to_numpy(float)
        - r8_rows.global_rmse.to_numpy(float)
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "vista-smst-paired-frontier-r4-r6-r8-v1",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.22), gridspec_kw={"width_ratios": [1.24, 1.0, 1.05]})
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="both", color=COLORS["grid"], linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)

    axis = axes[0]
    for key, label, _ in variants:
        x, y = ecdf(deltas[key])
        axis.step(x, y, where="post", color=COLORS[key], linewidth=1.25, label=label)
    axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    bound = max(abs(np.quantile(np.concatenate(list(deltas.values())), 0.01)), abs(np.quantile(np.concatenate(list(deltas.values())), 0.99)))
    axis.set_xlim(-1.08 * bound, 1.08 * bound)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Committed output $-$ global RMSE")
    axis.set_ylabel("Fraction of 3,884 R4 final locations")
    axis.set_title("(a) R4 paired error distributions", loc="left", fontweight="bold")
    axis.legend(loc="lower right", frameon=True, framealpha=0.94, borderpad=0.26, labelspacing=0.22, handlelength=1.3)

    axis = axes[1]
    for key, label, _ in variants:
        item = r6_summary["variants"][key]
        x = 100.0 * float(item["coverage"])
        y = -float(item["delta_rmse_vs_global"])
        changes = int(item["frozen_decoder_label_changes_vs_global"])
        harms = int(item["component_union_harms_vs_global"])
        marker = "o" if changes == 0 else "X"
        size = 26 + 0.24 * harms
        axis.scatter(x, y, s=size, marker=marker, color=COLORS[key], edgecolor="black", linewidth=0.45, zorder=3)
        display = {"unconstrained": "Unconstr.", "label_only": "Top label", "margin_only": "Margin only", "both": "SMST conjunction"}[key]
        offset = {"unconstrained": (-95, -2), "label_only": (-30, 12), "margin_only": (-23, -17), "both": (5, -14)}[key]
        axis.annotate(display, (x, y), xytext=offset, textcoords="offset points", fontsize=5.8, color=COLORS["ink"])
    axis.set_xlim(65, 103)
    axis.set_ylim(0.050, 0.0645)
    axis.set_xlabel("Residual commitment (%)")
    axis.set_ylabel("Mean global $-$ output RMSE")
    axis.set_title("(b) Fixed-branch trade-off", loc="left", fontweight="bold")

    axis = axes[2]
    x, y = ecdf(r8_delta)
    axis.step(x, y, where="post", color=COLORS["external"], linewidth=1.35)
    axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    r8_smst = r8_summary["selectors_external"]["smst_top_label_and_raw_margin"]
    low, high = r8_smst["spatial_cell_bootstrap"]["delta_rmse"]["ci95"]
    axis.axvspan(low, high, color=COLORS["external"], alpha=0.13, lw=0)
    axis.set_xlim(np.quantile(r8_delta, 0.005) - 0.05, np.quantile(r8_delta, 0.995) + 0.05)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("SMST $-$ frozen-global RMSE")
    axis.set_ylabel("Fraction of 1,490 locations")
    axis.set_title("(c) Frozen 2020 temporal check", loc="left", fontweight="bold")
    axis.text(0.03, 0.96, f"Mean $\\Delta$RMSE = {r8_smst['delta_rmse_vs_global']:+.4f}\ncell bootstrap 95% [{low:+.4f}, {high:+.4f}]\n450/1,490 commits; 164 declared harms", transform=axis.transAxes, va="top", ha="left", fontsize=5.9, bbox={"boxstyle": "round,pad=.23", "facecolor": "white", "edgecolor": "#999999", "linewidth": 0.45})

    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.25, top=0.88, wspace=0.42)
    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / "fig_smst_paired_frontier"
    metadata = {"Creator": "VISTA deterministic R4/R6/R8 paired-frontier builder", "CreationDate": None, "ModDate": None}
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015, metadata=metadata)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.015, metadata={"Date": None})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    contract = {
        "schema": "smst-paired-frontier-figure-contract-v1",
        "inputs": {key: digest(getattr(args, key.replace("-", "_"))) for key in ["r6-summary", "r6-per-location", "r8-summary", "r8-per-location"]},
        "semantic_contract": {
            "panel_a": "Post-R4 fixed-branch paired RMSE distributions; not a fresh primary test.",
            "panel_b": "Coverage--fidelity trade-off with decoder-label-change and declared-harm accounting.",
            "panel_c": "Frozen 2020 temporal check; positive delta is an unfavorable SMST RMSE result.",
        },
        "r4_rows": int(len(r6_rows)),
        "r8_rows": int(len(r8_rows)),
        "r8_delta_rmse_mean": float(r8_smst["delta_rmse_vs_global"]),
        "r8_delta_rmse_ci95": [float(low), float(high)],
    }
    (args.outdir / "fig_smst_paired_frontier_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = sorted(path for path in args.outdir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.outdir / "SHA256SUMS").write_text("\n".join(f"{digest(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
