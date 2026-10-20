"""Reconstruct one selected VISTA fold model; no reselection or controls.

The logistic classifier is fitted to convergence (Newton-Cholesky, tol 1e-10), as in
scripts/run_vista_comparison.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import vista_converged  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desis", type=Path, required=True)
    parser.add_argument("--prisma", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).parent
    manifest = json.loads((here / "selected_folds.json").read_text(encoding="utf-8"))
    decision = manifest["folds"][args.fold]
    original = vista_converged.load(here / "vendor" / "original_common_utility.py")
    if original.sha256(args.desis) != manifest["data_hashes"]["desis"] or original.sha256(args.prisma) != manifest["data_hashes"]["prisma"]:
        raise ValueError("Inputs differ from the declared 2021 release")
    data = original.load_release(args.desis, args.prisma)
    dev = original.make_partitions(data["lon"], data["lat"])[args.fold]["dev"]
    base = decision["stage1"]
    g = original.fit_diag_affine(data["x"][dev], data["y"][dev], base["global_alpha"])
    decoder = original.fit_logistic(data["y"][dev], data["class_idx"][dev], base["decoder_alpha"])
    gd = original.apply_diag(data["x"][dev], g)
    residual = original.fit_weighted_residual(data["x"][dev], data["y"][dev], gd,
                                              decoder.proba(gd), decision["original"]["residual_alpha"])
    np.savez_compressed(args.output, G_slope=g[0], G_intercept=g[1],
        decoder_mean=decoder.mean, decoder_scale=decoder.scale,
        decoder_coef=decoder.model.coef_, decoder_intercept=decoder.model.intercept_,
        V_slope=np.array([s[0] for s in residual]), V_intercept=np.array([s[1] for s in residual]),
        V_scale=decision["original"]["scale"], wavelength_nm=data["waves"], classes=original.CLASSES)


if __name__ == "__main__":
    main()
