"""Check supplementary diagnostics using replayed spectra and saved rows; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "records/results"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, name):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"{name}: keys differ")
        for key in expected:
            compare(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, int):
        require(actual == expected, f"{name}: count differs")
    else:
        require(np.allclose(actual, expected, atol=1e-9, rtol=0),
                f"{name}: numerical values differ")


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2, axis=1))


def sam(x, y):
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    cosine = np.divide((x * y).sum(axis=1), denominator,
                       out=np.ones(len(x)), where=denominator > 0)
    return np.arccos(np.clip(cosine, -1, 1))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    replay = args.replay_dir
    with np.load(replay / "single_split_2021_spectra_from_original_state.npz") as archive:
        z = {key: archive[key] for key in archive.files}
    with np.load(replay / "original_single_split_selected_state.npz") as archive:
        state = {key: archive[key] for key in archive.files}
    rows = pd.read_csv(replay / "single_split_2021_per_location_verified.csv")
    require(np.array_equal(rows.loc_id, z["loc_id"]), "Single-split row order differs")
    classes, truth = z["classes"], rows.crop.to_numpy()
    y, global_spectra, candidate = z["desis_reference"], z["fallback"], z["candidate"]

    def predict(spectra):
        logits = ((spectra - state["decoder_mean"]) / state["decoder_scale"]) @ state["decoder_coef"].T
        logits += state["decoder_intercept"]
        return classes[np.argsort(logits, axis=1)[:, -1]]

    def metrics(spectra, predicted=None):
        predicted = predict(spectra) if predicted is None else predicted
        per_class = f1_score(truth, predicted, labels=classes, average=None, zero_division=0)
        return {"accuracy": float(np.mean(predicted == truth)),
                "macro_f1": float(per_class.mean()),
                "per_class_f1": dict(zip(classes.tolist(), per_class.tolist())),
                "rmse": float(rmse(spectra, y).mean()),
                "sam_rad": float(sam(spectra, y).mean())}

    truth_index = np.array([np.flatnonzero(classes == label)[0] for label in truth])
    oracle = global_spectra + z["prisma_common59"] * state["residual_a"][truth_index] + state["residual_b"][truth_index]
    outputs = {"identity": z["prisma_common59"], "global_ridge": global_spectra,
               "unconstrained_semantic": candidate, "smst": z["returned"],
               "label_conditioned_oracle_diagnostic": oracle}
    single_metrics = {name: metrics(spectra) for name, spectra in outputs.items()}
    compare(single_metrics, load(RECORDS / "vista_smst_r4_20260812/final/FINAL_SUMMARY.json")["metrics"],
            "single-split five outputs")

    gpred, cpred = predict(global_spectra), predict(candidate)
    gerr, gsam = rmse(global_spectra, y), sam(global_spectra, y)
    margin_global = np.sort(z["fallback_logits"], axis=1)[:, -1] - np.sort(z["fallback_logits"], axis=1)[:, -2]
    margin_candidate = np.sort(z["candidate_logits"], axis=1)[:, -1] - np.sort(z["candidate_logits"], axis=1)[:, -2]
    masks = {"unconstrained": np.ones(len(y), dtype=bool),
             "label_only": cpred == gpred, "margin_only": margin_candidate >= margin_global,
             "both": (cpred == gpred) & (margin_candidate >= margin_global)}
    r6 = load(RECORDS / "vista_smst_r6_gate_components_20260812/run/R6_SUMMARY.json")
    r6rows = pd.read_csv(RECORDS / "vista_smst_r6_gate_components_20260812/run/R6_PER_LOCATION.csv")
    require(np.array_equal(r6rows.loc_id, z["loc_id"]), "S8 location order differs")
    variants = {}
    for name, mask in masks.items():
        require(np.array_equal(mask, r6rows[f"accept_{name}"]), f"S8 {name}: acceptance differs")
        spectra = np.where(mask[:, None], candidate, global_spectra)
        predicted = np.where(mask, cpred, gpred)
        values = metrics(spectra, predicted)
        values.pop("per_class_f1")
        delta = rmse(spectra, y) - gerr
        harm = (delta > .25) & (delta / np.maximum(gerr, 1e-12) > .05)
        wrong = (gpred == truth) & (predicted != truth)
        values.update(coverage=float(mask.mean()),
                      frozen_decoder_label_changes_vs_global=int((predicted != gpred).sum()),
                      correct_to_incorrect_vs_global=int(wrong.sum()),
                      spectral_harms_vs_global=int(harm.sum()),
                      component_union_harms_vs_global=int((harm | wrong).sum()),
                      delta_rmse_vs_global=float(delta.mean()),
                      delta_sam_rad_vs_global=float((sam(spectra, y) - gsam).mean()),
                      delta_macro_f1_vs_global=values["macro_f1"] - single_metrics["global_ridge"]["macro_f1"])
        variants[name] = values
    compare(variants, r6["variants"], "S8 four gate variants")

    errors_g, errors_o = global_spectra - y, z["returned"] - y
    accepted_errors = (candidate - y)[z["retained"]]
    band_values = {"wavelength_nm": z["wavelength_nm"],
                   "global_bias": errors_g.mean(axis=0),
                   "global_rmse": np.sqrt((errors_g ** 2).mean(axis=0)),
                   "smst_bias": errors_o.mean(axis=0),
                   "smst_rmse": np.sqrt((errors_o ** 2).mean(axis=0)),
                   "accepted_semantic_rmse": np.sqrt((accepted_errors ** 2).mean(axis=0)),
                   "accepted_semantic_bias": accepted_errors.mean(axis=0)}
    band_values["smst_minus_global_rmse"] = band_values["smst_rmse"] - band_values["global_rmse"]
    bands = pd.read_csv(RECORDS / "vista_smst_r4_wavelengthdiag_20260812/R4_WAVELENGTH_DIAGNOSTICS.csv")
    for name, actual in band_values.items():
        compare(actual, bands[name].to_numpy(), f"2021 wavelengths: {name}")
    with np.load(replay / "frozen_2020_spectra_from_original_state.npz") as transfer:
        bands20 = pd.read_csv(RECORDS / "vista_smst_r8_external_2020_20260812_attempt3/R8_WAVELENGTH_DIAGNOSTICS.csv")
        compare(transfer["wavelength_nm"], bands20.wavelength_nm.to_numpy(), "2020 wavelengths")
        # The archived diagonal_affine column uses a separately selected baseline,
        # not this frozen VISTA model's global map. That baseline state was not saved.
        for key, prefix in (("returned", "smst"),):
            error = transfer[key] - transfer["desis_reference"]
            compare(error.mean(axis=0), bands20[f"{prefix}_bias"].to_numpy(), f"2020 {prefix} bias")
            compare(np.sqrt((error ** 2).mean(axis=0)), bands20[f"{prefix}_rmse"].to_numpy(),
                    f"2020 {prefix} RMSE")

    geographic = pd.read_csv(RECORDS / "vista_smst_r5_regional_20260812/run/R5_PER_LOCATION.csv")
    geo_summary = load(RECORDS / "vista_smst_r5_regional_20260812/run/R5_SUMMARY.json")
    geo_counts = {}
    for direction, group in geographic.groupby("direction"):
        expected = geo_summary["directions"][direction]
        truth_geo = group.crop_label.to_numpy()
        before = group.global_prediction.to_numpy()
        after = np.where(group.smst_committed_semantic, group.semantic_prediction, before)
        for name, prediction in (("global_ridge", before), ("smst", after)):
            compare(float(np.mean(prediction == truth_geo)), expected["metrics"][name]["accuracy"],
                    f"{direction} {name} accuracy")
            compare(float(f1_score(truth_geo, prediction, labels=classes, average="macro", zero_division=0)),
                    expected["metrics"][name]["macro_f1"], f"{direction} {name} F1")
        delta = group.smst_rmse.to_numpy() - group.global_rmse.to_numpy()
        harms = (delta > .25) & (delta / np.maximum(group.global_rmse.to_numpy(), 1e-12) > .05)
        compare(int(harms.sum()), expected["paired_vs_global"]["spectral_harms_vs_global"],
                f"{direction} error count")
        compare(float(group.smst_committed_semantic.mean()), expected["paired_vs_global"]["coverage"],
                f"{direction} adoption")
        geo_counts[direction] = {"rows": len(group), "large_error_rows": int(harms.sum())}
    report = {"status": "PASS", "single_split_outputs": 5, "S8_gate_variants": 4,
              "single_split_rows": len(y), "wavelengths_checked_per_year": len(bands),
              "years": [2021, 2020], "geographic_saved_rows": geo_counts,
              "2020_wavelength_scope": "VISTA returned spectra; separate diagonal baseline state unavailable",
              "fitting_performed": False}
    (replay / "saved_diagnostics_verified.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
