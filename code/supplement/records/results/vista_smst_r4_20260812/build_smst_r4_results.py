#!/usr/bin/env python3
"""Deterministic publication figure for the locked SMST-R4 final result."""

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


PALETTE = {
    "blue_main": "#0F4D92",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral": "#CFCECE",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.1,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "smst-r4-20260812",
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--per-case", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = pd.read_csv(args.per_case)
    m = summary["metrics"]
    names = ["Id.", "Global", "Mix", "SMST", "Oracle"]
    keys = ["identity", "global_ridge", "unconstrained_semantic", "smst", "label_conditioned_oracle_diagnostic"]
    colors = [PALETTE["neutral"], PALETTE["red_strong"], PALETTE["violet"], PALETTE["blue_main"], PALETTE["green_3"]]
    hatches = ["", "", "//", "", ".."]
    rmse = [m[k]["rmse"] for k in keys]
    sam = [m[k]["sam_rad"] for k in keys]
    changed = int((rows.semantic_prediction != rows.global_prediction).sum())
    committed = int(rows.smst_committed_semantic.astype(bool).sum())
    coverage = float(rows.smst_committed_semantic.astype(bool).mean())
    harms = int(rows.union_harm_vs_global.astype(bool).sum())

    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.10), gridspec_kw={"width_ratios": [1.08, 1.08, 0.94]})
    x = np.arange(len(names))
    for ax, vals, ylabel, fmt in [(axes[0], rmse, "DESIS-reference RMSE\n(reflectance percentage points)", "{:.2f}"), (axes[1], sam, "Spectral angle (rad)", "{:.3f}")]:
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.55)
        for bar, hatch in zip(bars, hatches, strict=True): bar.set_hatch(hatch)
        ax.set_xticks(x, names)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.23)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.035 * max(vals), fmt.format(v), ha="center", va="bottom", fontsize=6.1, rotation=90)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("(a) Paired spectral error", loc="left", fontweight="bold")
    axes[1].set_title("(b) Angular discrepancy", loc="left", fontweight="bold")

    ax = axes[2]
    bx = np.arange(2)
    state_bars = ax.bar(bx, [changed, 0], color=[PALETTE["violet"], PALETTE["blue_main"]], edgecolor="black", linewidth=0.6, hatch=["//", ""])
    ax.set_xticks(bx, ["Unconstrained\nsemantic", "SMST\n(committed)"])
    ax.set_ylabel("Changed frozen-decoder\nlabels vs. global ridge")
    ax.set_ylim(0, max(330, changed * 1.22))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    for bar, value in zip(state_bars, [changed, 0], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(8, changed * .035), str(value), ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax.text(0.02, 0.98, f"SMST commits {committed}/{len(rows)} ({coverage:.1%})\n{harms} declared spectral harms vs. global ridge", transform=ax.transAxes, va="top", ha="left", fontsize=6.8, bbox={"boxstyle": "round,pad=.28", "facecolor": "white", "edgecolor": "#777777", "linewidth": .5})
    ax.set_title("(c) Semantic commitment", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.075, right=.992, bottom=.24, top=.88, wspace=.42)
    base = out / "fig_smst_r4_paired_results"
    metadata = {"Creator": "VISTA deterministic SMST-R4 figure builder", "CreationDate": None, "ModDate": None}
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.015, metadata=metadata)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=.015, metadata={"Date": None})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.015)
    plt.close(fig)
    contract = {"schema": "smst-r4-figure-contract-v1", "summary_sha256": sha256(args.summary), "per_case_sha256": sha256(args.per_case), "methods": keys, "semantic_contract": "Panels (a,b) compare the locked paired spectral metrics. Panel (c) reports frozen-decoder label changes relative to global ridge; SMST preserves that decoder label by its commit rule and does not claim true-state or population safety.", "values": {"rmse": rmse, "sam_rad": sam, "unconstrained_label_changes": changed, "smst_coverage": coverage, "smst_declared_spectral_harms": harms}}
    (out / "fig_smst_r4_paired_results_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
