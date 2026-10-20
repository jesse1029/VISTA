#!/usr/bin/env python3
"""Deterministic R4/R5 heterogeneity figure (no raw imagery)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "svg.hashsalt": "vista-smst-r4-r5-heterogeneity-v1",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"blue": "#1F5A9D", "purple": "#97519A", "green": "#77B86A", "red": "#BE4B49", "ink": "#202124"}
ORDER = ["alfalfa", "almonds", "corn", "grapes", "pistachios"]
SHORT = ["Alfalfa", "Almond", "Corn", "Grape", "Pistachio"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r4-csv", type=Path, required=True)
    parser.add_argument("--r5-summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.r4_csv.open(newline="", encoding="utf-8")))
    r5 = json.loads(args.r5_summary.read_text(encoding="utf-8"))
    deltas = {crop: [] for crop in ORDER}
    coverage, harms = [], []
    for crop in ORDER:
        group = [r for r in rows if r["crop_label"] == crop]
        deltas[crop] = [float(r["smst_rmse"]) - float(r["global_rmse"]) for r in group]
        coverage.append(100.0 * sum(r["smst_committed_semantic"] == "True" for r in group) / len(group))
        harms.append(100.0 * sum(r["spectral_harm_vs_global"] == "True" for r in group) / len(group))
    directions = ["west_to_east", "east_to_west"]
    dlabels = ["West$\\rightarrow$East", "East$\\rightarrow$West"]
    gains = [-r5["directions"][d]["paired_vs_global"]["smst_minus_global_rmse"] for d in directions]
    dcoverage = [100.0 * r5["directions"][d]["paired_vs_global"]["coverage"] for d in directions]
    dharm = [100.0 * r5["directions"][d]["paired_vs_global"]["spectral_harms_vs_global"] / r5["directions"][d]["test_rows"] for d in directions]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1.20, 1.05, .95]})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.1)
        ax.grid(axis="y", color="#D7D7D7", linewidth=.55, zorder=0)

    ax = axes[0]
    box = ax.boxplot([deltas[c] for c in ORDER], patch_artist=True, showfliers=False, widths=.58,
                     medianprops={"color": COLORS["ink"], "linewidth": 1.1},
                     whiskerprops={"color": COLORS["ink"], "linewidth": .8},
                     capprops={"color": COLORS["ink"], "linewidth": .8})
    for patch in box["boxes"]:
        patch.set(facecolor="#D6EAF8", edgecolor=COLORS["blue"], linewidth=1.0)
    ax.axhline(0, color="#666", linewidth=.8)
    ax.axhline(.25, color=COLORS["red"], linewidth=.9, linestyle="--")
    ax.text(5.35, .27, "+0.25", fontsize=5.7, color=COLORS["red"], ha="right")
    ax.set_xticks(range(1, 6), SHORT, rotation=24, ha="right")
    ax.set_ylabel("SMST $-$ global RMSE", fontsize=6.6)
    ax.set_title("(a) Final paired error by crop", fontsize=7.4, fontweight="bold", pad=3)

    ax = axes[1]
    x = np.arange(len(ORDER)); w = .36
    ax.bar(x - w / 2, coverage, width=w, color=COLORS["blue"], label="residual commit", zorder=2)
    ax.bar(x + w / 2, harms, width=w, color=COLORS["red"], label="declared harm", zorder=2)
    ax.set_xticks(x, SHORT, rotation=24, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Final locations (%)", fontsize=6.6)
    ax.set_title("(b) Gate use and harms", fontsize=7.4, fontweight="bold", pad=3)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", handlelength=1.0, borderpad=.12, labelspacing=.22)

    ax = axes[2]
    x = np.arange(2)
    ax.bar(x, gains, width=.52, color=COLORS["green"], zorder=2)
    for xi, gain, cov, harm in zip(x, gains, dcoverage, dharm, strict=True):
        ax.text(xi, gain + .0017, f"{gain:.3f}", ha="center", va="bottom", fontsize=5.9, color=COLORS["ink"])
        ax.text(xi, .0022, f"cov. {cov:.1f}%\nharm {harm:.1f}%", ha="center", va="bottom", fontsize=5.5, color=COLORS["ink"])
    ax.set_xticks(x, dlabels, rotation=18, ha="right")
    ax.set_ylabel("Global $-$ SMST RMSE", fontsize=6.6)
    ax.set_ylim(0, max(gains) * 1.28)
    ax.set_title("(c) Same-release regional stress", fontsize=7.4, fontweight="bold", pad=3)
    ax.text(.5, -.33, "Fixed R4 configuration; not independent data", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.35, color="#666")

    fig.subplots_adjust(left=.075, right=.992, top=.88, bottom=.29, wspace=.43)
    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / "fig_smst_r4_r5_heterogeneity"
    for suffix, kwargs in ((".pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
                           (".svg", {"metadata": {"Date": None}}),
                           (".png", {"dpi": 240})):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", pad_inches=.025, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
