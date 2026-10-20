#!/usr/bin/env python3
"""Render the immutable R4 validation grid in warning-free fixed-size LaTeX parts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def scientific(value: str) -> str:
    coefficient, exponent = value.split("e")
    if coefficient == "1":
        return "$10^{" + str(int(exponent)) + "}$"
    return "$" + coefficient + r"\!\times\!10^{" + str(int(exponent)) + "}$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 192:
        raise RuntimeError(f"Expected 192 R4 validation rows, found {len(rows)}")
    chunks = [rows[index : index + 32] for index in range(0, len(rows), 32)]
    if len(chunks) != 6 or any(len(chunk) != 32 for chunk in chunks):
        raise RuntimeError("Expected six 32-row validation-grid parts")

    header = r"$\lambda_g$ & $\lambda_f$ & $\lambda_r$ & $\delta$ & Macro-F1 & RMSE & SAM (rad) & Coverage \\\\"
    output: list[str] = []
    for part, chunk in enumerate(chunks, start=1):
        label = "tab:supp-validation-grid" if part == 1 else f"tab:supp-validation-grid-part{part}"
        output.extend(
            [
                r"\clearpage",
                r"\refstepcounter{table}",
                rf"\label{{{label}}}",
                r"\begingroup",
                r"\footnotesize",
                r"\begin{center}",
                rf"\textbf{{\tablename~\thetable.}} Complete immutable R4 validation grid (192 operating points), part {part} of 6. The highlighted row is the locked R4 choice; all rows precede any R4 final evaluation.",
                r"\renewcommand{\arraystretch}{0.94}",
                r"\begin{tabular}{rrrrrrrr}",
                r"\toprule",
                header,
                r"\midrule",
            ]
        )
        for row in chunk:
            params = dict(piece.split("=") for piece in row["model_key"].split("|"))
            selected = row["model_key"] == "g=1e-02|d=1e-01|r=1e-04" and float(row["margin"]) == 0.0
            if selected:
                output.append(r"\rowcolor{vistabg}")
            output.append(
                "{} & {} & {} & {:.2f} & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\".format(
                    scientific(params["g"]),
                    scientific(params["d"]),
                    scientific(params["r"]),
                    float(row["margin"]),
                    float(row["macro_f1"]),
                    float(row["rmse"]),
                    float(row["sam_rad"]),
                    float(row["coverage"]),
                )
            )
        output.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\endgroup", ""])
    args.output.write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    main()
