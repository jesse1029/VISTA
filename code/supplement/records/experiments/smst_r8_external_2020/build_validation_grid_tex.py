#!/usr/bin/env python3
"""Render the immutable R4 validation grid as a readable supplement table."""

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
    out = [
        r"\begin{longtable}{rrrrrrrr}",
        r"\caption{Complete immutable R4 validation grid (192 operating points). The highlighted row is the locked R4 choice. All rows precede any R4 final evaluation.} \label{tab:supp-validation-grid}\\",
        r"\toprule",
        r"$\lambda_g$ & $\lambda_f$ & $\lambda_r$ & $\delta$ & Macro-F1 & RMSE & SAM (rad) & Coverage \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{c}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"$\lambda_g$ & $\lambda_f$ & $\lambda_r$ & $\delta$ & Macro-F1 & RMSE & SAM (rad) & Coverage \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        parts = dict(piece.split("=") for piece in row["model_key"].split("|"))
        selected = row["model_key"] == "g=1e-02|d=1e-01|r=1e-04" and float(row["margin"]) == 0.0
        if selected:
            out.append(r"\rowcolor{vistabg}")
        out.append(
            "{} & {} & {} & {:.2f} & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\".format(
                scientific(parts["g"]), scientific(parts["d"]), scientific(parts["r"]),
                float(row["margin"]), float(row["macro_f1"]), float(row["rmse"]),
                float(row["sam_rad"]), float(row["coverage"])
            )
        )
    out.extend([r"\bottomrule", r"\end{longtable}", ""])
    args.output.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
