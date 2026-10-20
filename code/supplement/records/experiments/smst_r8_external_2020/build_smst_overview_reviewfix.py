#!/usr/bin/env python3
"""Deterministically render the SMST-R4 method overview as SVG, PDF, and PNG.

The drawing is intentionally schematic: it records the data roles and the
commit rule without depicting spectra as natural imagery.  Values shown in the
evaluation panel are read from the immutable R4 final summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "svg.hashsalt": "vista-smst-r4-overview-v1",
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "source": "#D6EAF8",
    "target": "#FDEBD0",
    "model": "#D5F5E3",
    "candidate": "#E8DAEF",
    "gate": "#D4E6F1",
    "commit": "#D5F5E3",
    "fallback": "#F2F3F4",
    "eval": "#FCF3CF",
    "ink": "#202124",
    "muted": "#5F6368",
}


def add_box(ax, xy, wh, title, body, color, *, edge="#39424E", title_size=10.6,
            body_size=8.5, rounded=0.07, dashed=False):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.018,rounding_size={rounded}",
        facecolor=color, edgecolor=edge, linewidth=1.35,
        linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.026, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=COLORS["ink"], zorder=3)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
            fontsize=body_size, color=COLORS["ink"], linespacing=1.22, zorder=3)
    return box


def arrow(ax, start, end, *, label=None, color="#39424E", label_offset=(0.0, 0.0),
          style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                                 linewidth=1.10, shrinkA=1.5, shrinkB=8.0,
                                 color=color, zorder=1))
    if label:
        ax.text((start[0] + end[0]) / 2 + label_offset[0],
                (start[1] + end[1]) / 2 + label_offset[1], label,
                ha="center", va="center", fontsize=11.5, color=COLORS["muted"],
                bbox={"boxstyle": "round,pad=0.13", "facecolor": "white", "edgecolor": "none", "alpha": .95},
                zorder=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    n_final = int(summary["final_row_count"])
    paired = summary["paired_vs_global"]
    coverage = 100.0 * float(paired["coverage"])
    committed = round(float(paired["coverage"]) * n_final)
    rmse_gain = -float(paired["smst_minus_global_rmse"])
    sam_gain = -float(paired["smst_minus_global_sam_rad"])
    label_changes = int(paired["semantic_label_changes_before_gate"])
    harms = int(paired["spectral_harms_vs_global"])

    # All typography is set at the final two-column physical width (6.9 in).
    fig, ax = plt.subplots(figsize=(6.9, 3.75), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.965, "Semantic-margin spectral transport (SMST)", fontsize=10.8,
            fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.02, 0.910, "Development and validation lock the transport; deployment commits a residual only when the frozen decision is preserved.",
            fontsize=6.3, color=COLORS["muted"], va="top")

    add_box(ax, (0.02, 0.59), (0.20, 0.22), "Paired development",
            "DESIS ref. + source labels\nPRISMA target; 59 common bands",
            COLORS["source"], title_size=7.3, body_size=5.35, rounded=0.025)
    add_box(ax, (0.28, 0.59), (0.18, 0.22), "Development models",
            r"train $f$; fit $T_g$" + "\n" + "and residual maps",
            COLORS["candidate"], title_size=6.8, body_size=5.85, rounded=0.025)
    add_box(ax, (0.51, 0.59), (0.11, 0.22), "Candidates",
            r"$T_g(z)$" + "\n" + r"$T_s(z)$", COLORS["candidate"], title_size=7.1, body_size=6.6, rounded=0.025)
    add_box(ax, (0.675, 0.53), (0.14, 0.30), "Frozen gate",
            "same top-1" + "\n" + "nondec. margin",
            COLORS["gate"], title_size=6.6, body_size=5.8, rounded=0.025)
    add_box(ax, (0.855, 0.69), (0.125, 0.15), "Commit", r"$T_s(z)$", COLORS["commit"],
            title_size=6.6, body_size=6.6, rounded=0.025)
    add_box(ax, (0.855, 0.51), (0.125, 0.15), "Fallback", r"$T_g(z)$", COLORS["fallback"],
            title_size=6.45, body_size=6.6, rounded=0.025)
    arrow(ax, (0.22, 0.70), (0.28, 0.70))
    arrow(ax, (0.46, 0.70), (0.51, 0.70))
    arrow(ax, (0.62, 0.70), (0.675, 0.70))
    arrow(ax, (0.815, 0.75), (0.855, 0.765), color="#18864B")
    arrow(ax, (0.815, 0.62), (0.855, 0.585), color="#6B7280")

    ax.plot([0.02, .98], [.46, .46], color="#8A6D3B", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(.50, .487, "validation labels + paired spectra select", ha="center", va="bottom",
            fontsize=5.35, color="#7C5C18")
    ax.text(.50, .463, "lock before opening the 0.01°-cell final split", ha="center", va="bottom",
            fontsize=5.35, color="#7C5C18")
    add_box(ax, (0.02, 0.16), (0.28, 0.20), "Locked paired final split",
            f"matched PRISMA--DESIS\n{n_final} hash-held locations",
            COLORS["eval"], title_size=7.0, body_size=6.0, rounded=0.025)
    add_box(ax, (0.37, 0.13), (0.61, 0.24), "Observed paired outcome",
            rf"Unconstrained $T_s$: {label_changes} top-label changes vs. $T_g$; SMST: 0" + "\n"
            + rf"{committed}/{n_final} residual commits ({coverage:.1f}\%)" + "\n"
            + rf"vs. $T_g$: RMSE $\downarrow$ {rmse_gain:.4f}, SAM $\downarrow$ {sam_gain:.4f} rad; {harms} declared paired-RMSE harms",
            COLORS["eval"], title_size=7.2, body_size=5.25, rounded=0.025)
    arrow(ax, (0.30, .26), (.37, .26))
    ax.text(.02, .072, "DESIS source labels train $f$; validation labels select the locked configuration; final labels score decoder predictions.",
            fontsize=5.25, color=COLORS["muted"])
    ax.text(.02, .035, "Paired DESIS spectra fit/select/evaluate transport; neither target labels nor paired references enter per-sample weighting or gating.",
            fontsize=5.05, color=COLORS["muted"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / "fig_smst_r4_overview"
    for suffix, kwargs in ((".pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
                           (".svg", {"metadata": {"Date": None}}),
                           (".png", {"dpi": 240})):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.03, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
