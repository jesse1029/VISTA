#!/usr/bin/env python3
"""Render one prespecified R4 validation-grid page in a fresh process."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def scientific(value: str) -> str:
    coefficient, exponent = value.split("e")
    if coefficient == "1":
        return f"10^{int(exponent)}"
    return f"{coefficient}e{int(exponent)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.part <= 6:
        raise RuntimeError("--part must be in 1..6")
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 192:
        raise RuntimeError(f"Expected 192 rows, found {len(rows)}")
    start = (args.part - 1) * 32
    chunk = rows[start : start + 32]
    if len(chunk) != 32:
        raise RuntimeError("Each page must contain 32 rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Complete prespecified R4 validation grid",
        "Author": "VISTA",
        "Subject": "All 192 pre-final validation operating points",
        "Creator": "VISTA deterministic validation-grid renderer",
        "CreationDate": None,
        "ModDate": None,
    }
    figure = plt.figure(figsize=(8.5, 11.0))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    figure.text(0.5, 0.962, "Complete prespecified R4 validation grid", ha="center", va="top", fontsize=13.2, fontweight="bold")
    figure.text(0.5, 0.935, f"All 192 operating points precede R4 final evaluation, part {args.part} of 6", ha="center", va="top", fontsize=9.4)
    columns = ["$\\lambda_g$", "$\\lambda_f$", "$\\lambda_r$", "$\\delta$", "Macro-F1", "RMSE", "SAM (rad)", "Coverage"]
    body: list[list[str]] = []
    selected_row: int | None = None
    for index, row in enumerate(chunk):
        params = dict(piece.split("=") for piece in row["model_key"].split("|"))
        body.append([
            scientific(params["g"]),
            scientific(params["d"]),
            scientific(params["r"]),
            f"{float(row['margin']):.2f}",
            f"{float(row['macro_f1']):.6f}",
            f"{float(row['rmse']):.6f}",
            f"{float(row['sam_rad']):.6f}",
            f"{float(row['coverage']):.6f}",
        ])
        if row["model_key"] == "g=1e-02|d=1e-01|r=1e-04" and float(row["margin"]) == 0.0:
            selected_row = index
    table = axis.table(cellText=body, colLabels=columns, cellLoc="right", colLoc="center", bbox=[0.085, 0.105, 0.83, 0.785], colWidths=[0.105, 0.105, 0.105, 0.07, 0.14, 0.14, 0.16, 0.145])
    table.auto_set_font_size(False)
    table.set_fontsize(8.45)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_linewidth(0.25 if row_index else 0.6)
        if row_index == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold", ha="center")
        elif selected_row is not None and row_index == selected_row + 1:
            cell.set_facecolor("#E6F0FA")
    figure.text(0.085, 0.067, "Highlighted row: locked R4 configuration. No final R4 location is read by this grid.", ha="left", va="bottom", fontsize=8.1)
    figure.text(0.915, 0.067, f"Page {args.part}/6", ha="right", va="bottom", fontsize=8.1)
    figure.savefig(args.output, format="pdf", bbox_inches=None, metadata=metadata)
    plt.close(figure)
    contract = {
        "schema": "smst-r4-validation-grid-single-page-v1",
        "input_sha256": digest(args.input),
        "output_sha256": digest(args.output),
        "part": args.part,
        "row_start": start,
        "row_count": len(chunk),
        "selected_model_key": "g=1e-02|d=1e-01|r=1e-04",
        "selected_margin": 0.0,
        "semantic_contract": "One fixed page of the complete prespecified validation grid. No model is fit or selected by this renderer.",
    }
    args.contract.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
