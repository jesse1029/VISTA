#!/usr/bin/env python3
"""Deterministic supplementary figure from locked R4 band-wise diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_csv(args.csv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if len(rows) != 59 or summary["wavelength_count"] != 59:
        raise RuntimeError("Expected the locked 59-band R4 diagnostic")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.8, "axes.titlesize": 8.6, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.8, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "vista-r4-wavelength-diagnostics-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.18))
    colors = {"global": "#B64342", "smst": "#0F4D92"}
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#D9D9D9", linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)
        axis.axhline(0, color="#555555", linewidth=0.65) if axis is axes[1] else None
    wavelength = rows.wavelength_nm
    axes[0].plot(wavelength, rows.global_rmse, color=colors["global"], linewidth=1.35, label="Global ridge")
    axes[0].plot(wavelength, rows.smst_rmse, color=colors["smst"], linewidth=1.35, label="SMST output")
    axes[0].set_xlabel("DESIS-reference wavelength (nm)")
    axes[0].set_ylabel("Band-wise RMSE")
    axes[0].set_title("(a) Paired spectral error", loc="left", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].plot(wavelength, rows.global_bias, color=colors["global"], linewidth=1.25, label="Global ridge")
    axes[1].plot(wavelength, rows.smst_bias, color=colors["smst"], linewidth=1.25, label="SMST output")
    axes[1].set_xlabel("DESIS-reference wavelength (nm)")
    axes[1].set_ylabel("Mean signed error")
    axes[1].set_title("(b) Paired signed bias", loc="left", fontweight="bold")
    axes[1].text(0.02, 0.03, f"Locked R4 final: {summary['final_rows']:,} locations\nSMST residual committed at {summary['accepted_rows']:,}", transform=axes[1].transAxes, fontsize=6.15, va="bottom", ha="left", bbox={"boxstyle": "round,pad=.22", "facecolor": "white", "edgecolor": "#999999", "linewidth": 0.45})
    fig.subplots_adjust(left=.08, right=.99, bottom=.25, top=.86, wspace=.32)
    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / "fig_r4_wavelength_diagnostics"
    metadata = {"Creator": "VISTA deterministic R4 wavelength diagnostic builder", "CreationDate": None, "ModDate": None}
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.015, metadata=metadata)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=.015, metadata={"Date": None})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.015)
    plt.close(fig)
    contract = {"schema": "smst-r4-wavelength-figure-contract-v1", "diagnostic_csv_sha256": sha256(args.csv), "diagnostic_summary_sha256": sha256(args.summary), "semantic_contract": "Post-R4, locked-output band-wise paired RMSE and signed-bias diagnostic. It does not refit a model or establish band-wise population effects."}
    (args.outdir / "fig_r4_wavelength_diagnostics_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = sorted(path for path in args.outdir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.outdir / "SHA256SUMS").write_text("\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
