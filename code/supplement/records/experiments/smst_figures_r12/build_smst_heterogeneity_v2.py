#!/usr/bin/env python3
"""Deterministic R4 heterogeneity and same-release R5 stress figure."""

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


ORDER = ["alfalfa", "almonds", "corn", "grapes", "pistachios"]
SHORT = ["Alfalfa", "Almond", "Corn", "Grape", "Pistachio"]
COLORS = {"blue": "#1F5A9D", "red": "#BE4B49", "green": "#5A9C54", "ink": "#202124", "grid": "#D8D8D8"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4-csv", type=Path, required=True)
    parser.add_argument("--r5-summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    r4 = pd.read_csv(args.r4_csv)
    r5 = json.loads(args.r5_summary.read_text(encoding="utf-8"))
    if sorted(r4.crop_label.unique()) != ORDER:
        raise RuntimeError("Unexpected R4 crop-label support")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "vista-smst-heterogeneity-r4-r5-v2",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.38), gridspec_kw={"width_ratios": [1.28, 1.16, 0.94]})
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)

    rng = np.random.default_rng(20260812)
    axis = axes[0]
    grouped = []
    counts = []
    for position, crop in enumerate(ORDER, start=1):
        values = (r4.loc[r4.crop_label == crop, "smst_rmse"] - r4.loc[r4.crop_label == crop, "global_rmse"]).to_numpy(float)
        grouped.append(values)
        counts.append(int(len(values)))
        jitter = rng.uniform(-0.22, 0.22, size=len(values))
        axis.scatter(np.full(len(values), position) + jitter, values, s=3.8, alpha=0.10, color=COLORS["blue"], edgecolors="none", rasterized=True, zorder=1)
    boxes = axis.boxplot(grouped, positions=np.arange(1, 6), widths=0.48, whis=(5, 95), showfliers=False, patch_artist=True, zorder=2, medianprops={"color": COLORS["ink"], "linewidth": 1.0}, whiskerprops={"color": COLORS["ink"], "linewidth": 0.75}, capprops={"color": COLORS["ink"], "linewidth": 0.75})
    for box in boxes["boxes"]:
        box.set(facecolor="#D8E8F6", edgecolor=COLORS["blue"], linewidth=0.9)
    axis.axhline(0, color="#555555", linewidth=0.75)
    axis.axhline(0.25, color=COLORS["red"], linewidth=0.85, linestyle="--")
    axis.text(5.38, 0.265, "+0.25 absolute component", fontsize=5.3, ha="right", color=COLORS["red"])
    axis.set_xticks(np.arange(1, 6), [f"{name}\nn={count:,}" for name, count in zip(SHORT, counts, strict=True)])
    axis.set_ylabel("SMST $-$ global RMSE")
    axis.set_title("(a) R4 final paired error by crop", loc="left", fontweight="bold")
    axis.text(0.01, 0.02, "Points: all final locations; boxes: IQR; whiskers: 5--95%", transform=axis.transAxes, fontsize=5.45, ha="left", va="bottom", bbox={"boxstyle": "round,pad=.18", "facecolor": "white", "edgecolor": "#999999", "linewidth": 0.4})

    axis = axes[1]
    commitment, harms, commitment_error, harms_error = [], [], [], []
    for crop in ORDER:
        frame = r4.loc[r4.crop_label == crop]
        total = len(frame)
        committed = int(frame.smst_committed_semantic.sum())
        declared_harms = int(frame.spectral_harm_vs_global.sum())
        commitment.append(100 * committed / total)
        harms.append(100 * declared_harms / total)
        low, high = wilson_interval(committed, total)
        commitment_error.append([max(0.0, 100 * (committed / total - low)), max(0.0, 100 * (high - committed / total))])
        low, high = wilson_interval(declared_harms, total)
        harms_error.append([max(0.0, 100 * (declared_harms / total - low)), max(0.0, 100 * (high - declared_harms / total))])
    x = np.arange(len(ORDER)); width = 0.36
    axis.bar(x - width / 2, commitment, width=width, yerr=np.array(commitment_error).T, capsize=2.0, color=COLORS["blue"], edgecolor="black", linewidth=0.4, label="residual commitment", zorder=2)
    axis.bar(x + width / 2, harms, width=width, yerr=np.array(harms_error).T, capsize=2.0, color=COLORS["red"], edgecolor="black", linewidth=0.4, label="declared harm", zorder=2)
    axis.set_xticks(x, SHORT, rotation=23, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("R4 final locations (%)")
    axis.set_title("(b) Commitment and declared harm", loc="left", fontweight="bold")
    axis.legend(loc="upper left", frameon=False, handlelength=1.1, borderpad=0.16, labelspacing=0.2)
    axis.text(0.01, 0.02, "Error bars: finite-sample Wilson 95% intervals", transform=axis.transAxes, fontsize=5.45, ha="left", va="bottom", bbox={"boxstyle": "round,pad=.16", "facecolor": "white", "edgecolor": "#999999", "linewidth": 0.4})

    axis = axes[2]
    directions = ["west_to_east", "east_to_west"]
    labels = ["West$\\rightarrow$East", "East$\\rightarrow$West"]
    gains = [-r5["directions"][direction]["paired_vs_global"]["smst_minus_global_rmse"] for direction in directions]
    bars = axis.bar(np.arange(2), gains, width=0.54, color=COLORS["green"], edgecolor="black", linewidth=0.5, zorder=2)
    for index, (bar, direction, gain) in enumerate(zip(bars, directions, gains, strict=True)):
        record = r5["directions"][direction]
        paired = record["paired_vs_global"]
        axis.text(bar.get_x() + bar.get_width() / 2, gain + 0.0015, f"{gain:.3f}", ha="center", va="bottom", fontsize=6.2, fontweight="bold")
        axis.text(bar.get_x() + bar.get_width() / 2, 0.002, f"n={record['test_rows']:,}\ncov. {100 * paired['coverage']:.1f}%\nharm {100 * paired['spectral_harms_vs_global'] / record['test_rows']:.1f}%", ha="center", va="bottom", fontsize=5.25)
    axis.set_xticks(np.arange(2), labels, rotation=18, ha="right")
    axis.set_ylim(0, max(gains) * 1.34)
    axis.set_ylabel("Global $-$ SMST RMSE")
    axis.set_title("(c) Same-release geographic stress", loc="left", fontweight="bold")
    axis.text(0.5, -0.39, "Fixed R4 configuration; not independent data", transform=axis.transAxes, ha="center", va="top", fontsize=5.4, color="#555555")

    fig.subplots_adjust(left=0.073, right=0.992, bottom=0.31, top=0.88, wspace=0.42)
    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / "fig_smst_r4_r5_heterogeneity"
    metadata = {"Creator": "VISTA deterministic R4/R5 heterogeneity builder", "CreationDate": None, "ModDate": None}
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015, metadata=metadata)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.015, metadata={"Date": None})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    contract = {
        "schema": "smst-r4-r5-heterogeneity-figure-contract-v2",
        "r4_per_location_sha256": digest(args.r4_csv),
        "r5_summary_sha256": digest(args.r5_summary),
        "semantic_contract": {
            "panel_a": "All R4 final locations plus finite sample summaries by released crop label.",
            "panel_b": "R4 finite-bank rates with Wilson intervals; no crop-effect claim.",
            "panel_c": "Fixed post-R4 same-release geographic stress; not independent validation.",
        },
        "r4_counts_by_crop": dict(zip(ORDER, counts, strict=True)),
    }
    (args.outdir / "fig_smst_r4_r5_heterogeneity_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = sorted(path for path in args.outdir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.outdir / "SHA256SUMS").write_text("\n".join(f"{digest(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
