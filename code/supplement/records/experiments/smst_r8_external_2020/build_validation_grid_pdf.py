#!/usr/bin/env python3
"""Render all 192 immutable R4 validation rows as six fixed vector-PDF pages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 192:
        raise RuntimeError(f"Expected 192 rows, found {len(rows)}")
    chunks = [rows[index : index + 32] for index in range(0, 192, 32)]
    if len(chunks) != 6:
        raise RuntimeError("Expected six validation-grid pages")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.part_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "Complete immutable R4 validation grid", "Author": "VISTA", "Subject": "All 192 pre-final validation operating points", "Creator": "VISTA deterministic validation-grid renderer", "CreationDate": None, "ModDate": None}
    with PdfPages(args.output, metadata=metadata) as pdf:
        for part, chunk in enumerate(chunks, start=1):
            figure = plt.figure(figsize=(8.5, 11.0))
            axis = figure.add_axes([0, 0, 1, 1])
            axis.axis("off")
            figure.text(0.5, 0.962, "Complete immutable R4 validation grid", ha="center", va="top", fontsize=13.2, fontweight="bold")
            figure.text(0.5, 0.935, f"All 192 operating points precede R4 final evaluation, part {part} of 6", ha="center", va="top", fontsize=9.4)
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
            for (row_index, column_index), cell in table.get_celld().items():
                cell.set_linewidth(0.25 if row_index else 0.6)
                if row_index == 0:
                    cell.set_facecolor("#F0F0F0")
                    cell.set_text_props(fontweight="bold", ha="center")
                elif selected_row is not None and row_index == selected_row + 1:
                    cell.set_facecolor("#E6F0FA")
            figure.text(0.085, 0.067, "Highlighted row: locked R4 configuration. No final R4 location is read by this grid.", ha="left", va="bottom", fontsize=8.1)
            figure.text(0.915, 0.067, f"Page {part}/6", ha="right", va="bottom", fontsize=8.1)
            pdf.savefig(figure, bbox_inches=None)
            plt.close(figure)
    subprocess.run(
        ["pdfseparate", str(args.output), str(args.part_dir / "r4_validation_grid_part%d.pdf")],
        check=True,
    )
    part_paths = [args.part_dir / f"r4_validation_grid_part{part}.pdf" for part in range(1, 7)]
    contract = {"schema": "smst-r4-validation-grid-fixed-pdf-v2", "input_sha256": digest(args.input), "pdf_sha256": digest(args.output), "part_pdf_sha256": {path.name: digest(path) for path in part_paths}, "row_count": len(rows), "page_count": 6, "selected_model_key": "g=1e-02|d=1e-01|r=1e-04", "selected_margin": 0.0, "semantic_contract": "Fixed, complete presentation of the pre-final immutable validation grid. No model is fit or selected by this renderer."}
    args.contract.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
