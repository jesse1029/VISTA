"""Replay saved VISTA models and recompute paper summaries without fitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_values(actual, expected, path="root"):
    """Compare scientific values, including row counts and nested intervals."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: keys differ")
        for key in expected:
            compare_values(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: lengths differ")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_values(a, b, f"{path}[{i}]")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        require(np.isclose(actual, expected, atol=1e-9, rtol=1e-10, equal_nan=False),
                f"{path}: {actual} != {expected}")
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with the five files listed in input_hashes.json")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args()
    require(not sys.flags.optimize,
            "Run verification without -O/-OO or PYTHONOPTIMIZE; all scientific checks must remain active")
    from check_manifest import check_release
    checked_files = check_release()
    data, out = args.data_dir.resolve(), args.out.resolve()
    require(not out.exists() or not any(out.iterdir()), "Choose a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    hashes = load_json(ROOT / "input_hashes.json")
    for name, expected in hashes.items():
        path = data / name
        require(path.is_file(), f"Missing input: {path}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Input SHA-256 mismatch: {name}")
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "input_hashes": hashes, "model_fitting_performed": False,
        "checks": {"release_integrity": {"files": checked_files, "all_hashes_match": True}},
    }

    # Import the existing loader, without entering its training CLI.
    spec = importlib.util.spec_from_file_location("original_common_utility", ROOT / "vendor/original_common_utility.py")
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    inputs = original.load_release(data / "desis_2021.csv", data / "prisma_2021.csv")
    np.savez_compressed(out / "principal_inputs.npz", prisma_common59=inputs["x"],
                        desis_reference=inputs["y"], wavelength_nm=inputs["waves"], loc_id=inputs["loc_id"])

    from vista import FrozenVISTA
    source = ROOT / "reported_results/converged_rerun"
    rows = pd.read_csv(source / "principal/heldout_per_location.csv").sort_values("row_index")
    require(np.array_equal(rows.row_index.to_numpy(), np.arange(16581)), "Principal row indices differ")
    require(np.array_equal(rows.loc_id.to_numpy(), inputs["loc_id"]), "Principal location order differs")
    require(np.array_equal(rows.class_idx.to_numpy(), inputs["class_idx"]), "Principal crop labels differ")
    folds, max_error, labels_changed = [], 0.0, 0
    reference_predictions = np.empty(len(rows), dtype=int)
    correction_predictions = np.empty(len(rows), dtype=int)
    parts = original.make_partitions(inputs["lon"], inputs["lat"])
    manifest = pd.read_csv(source / "principal/split_manifest.csv")
    require(np.array_equal(manifest.loc_id.to_numpy(), inputs["loc_id"]), "Split manifest locations differ")
    for part in parts:
        k = part["fold"]
        test = part["test"]
        role = np.full(len(rows), "excluded", dtype=object)
        for key, label in (("dev", "development"), ("val", "validation"), ("test", "test")):
            role[part[key]] = label
        require(np.array_equal(role, manifest[f"fold_{k}_role"].to_numpy()), f"Fold {k} partitions differ")
        model = FrozenVISTA.load(ROOT / f"models/fold_{k}.npz")
        pred = model.predict(inputs["x"][test])
        with np.load(source / f"principal/spectra_fold_{k}.npz", allow_pickle=False) as saved:
            require(np.array_equal(test, saved["row_index"]), f"Fold {k} held-out order differs")
            error = float(np.max(np.abs(pred.spectra - saved["logistic_partial"])))
            require(error < 1e-10, f"Fold {k} saved spectra differ: {error}")
            max_error = max(max_error, error)
        expected = rows.iloc[test]
        require(np.all(expected.fold.to_numpy() == k), f"Fold {k} row assignments differ")
        require(np.array_equal(pred.committed.astype(int), expected.logistic_partial__commit), f"Fold {k} adoption differs")
        rmse = np.sqrt(np.mean((pred.spectra - inputs["y"][test]) ** 2, axis=1))
        require(np.allclose(rmse, expected.logistic_partial__rmse, atol=1e-10, rtol=0), f"Fold {k} RMSE differs")
        returned_labels = model.probabilities(pred.spectra).argmax(axis=1)
        require(np.array_equal(pred.annotation_index, expected.logistic_global__pred),
                f"Fold {k} global classifier predictions differ")
        require(np.array_equal(returned_labels, expected.logistic_partial__pred),
                f"Fold {k} returned classifier predictions differ")
        labels_changed += int(np.sum(returned_labels != pred.annotation_index))
        reference_predictions[test] = model.probabilities(inputs["y"][test]).argmax(axis=1)
        correction_predictions[test] = model.probabilities(pred.candidate).argmax(axis=1)
        folds.append({"fold": k, "test_rows": len(test), "maximum_spectral_difference": error})
    require(labels_changed == 0, "The returned classifier labels changed")
    report["checks"]["principal_model_replay"] = {
        "rows": len(rows), "folds": folds, "maximum_spectral_difference": max_error,
        "classifier_labels_changed": labels_changed,
        "adopted_rows": int(rows.logistic_partial__commit.sum()),
        "improved_rows": int((rows.logistic_partial__delta_rmse < 0).sum()),
        "large_error_rows": int(rows.logistic_partial__tail.sum()),
    }
    def reference_summary(mask):
        true = inputs["class_idx"][mask]
        ref = reference_predictions[mask]
        prior = rows.logistic_global__pred.to_numpy()[mask]
        corrected = correction_predictions[mask]
        change = corrected != prior
        return {
            "n": int(mask.sum()), "ref_acc": float(np.mean(true == ref)),
            "ref_f1": float(f1_score(true, ref, labels=np.arange(5), average="macro", zero_division=0)),
            "agree": float(np.mean(prior == ref)), "changes": int(change.sum()),
            "toward": int(np.sum(change & (corrected == ref) & (prior != ref))),
            "away": int(np.sum(change & (prior == ref) & (corrected != ref))),
            "between": int(np.sum(change & (prior != ref) & (corrected != ref))),
        }
    anchor = {
        "overall": reference_summary(np.ones(len(rows), dtype=bool)),
        "crop": {crop: reference_summary(inputs["crop"] == crop) for crop in sorted(set(inputs["crop"]))},
    }
    compare_values(anchor, load_json(source / "reference_anchor.json"))
    report["checks"]["principal_reference_classifier"] = anchor
    additional = pd.read_csv(source / "additional/heldout_additional.csv").sort_values("row_index")
    require(np.array_equal(rows[["row_index", "loc_id", "fold"]], additional[["row_index", "loc_id", "fold"]]),
            "Additional-method locations or folds differ")
    combined = pd.concat([rows.reset_index(drop=True),
                          additional.drop(columns=["row_index", "loc_id", "fold"]).reset_index(drop=True)], axis=1)
    before = combined.logistic_global__pred.to_numpy()
    truth = combined.class_idx.to_numpy()
    transitions = {}
    for method in ("full_ridge", "pds", "pls", "krr", "mlp"):
        after = combined[f"{method}__pred"].to_numpy()
        wrong_to_right = (before != truth) & (after == truth)
        record = {
            "incorrect_to_correct": int(wrong_to_right.sum()),
            "correct_to_incorrect": int(((before == truth) & (after != truth)).sum()),
            "incorrect_class_changed": int(((before != truth) & (after != truth) & (before != after)).sum()),
            "classification_improved_but_rmse_increased": int(
                (wrong_to_right & (combined[f"{method}__delta_rmse"].to_numpy() > 0)).sum()),
        }
        transitions[method] = record
    # These independent counts are also written explicitly for the paper examples.
    require(transitions["full_ridge"]["incorrect_to_correct"] == 2044 and
            transitions["full_ridge"]["classification_improved_but_rmse_increased"] == 830,
            "The ridge example differs from the manuscript")
    report["checks"]["classification_transitions"] = transitions
    print("PASS: all 16,581 held-out spectra, five spatial folds and classifier labels", flush=True)

    # Existing summary programs expect the original base/ext directory names.
    run = out / "principal"
    shutil.copytree(source / "principal", run / "base")
    shutil.copytree(source / "additional", run / "ext")

    def command(name, arguments):
        print(f"Checking {name}...", flush=True)
        with (out / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True,
                           stdout=log, stderr=subprocess.STDOUT)

    command("paper_numbers", [ROOT / "scripts/paper_numbers.py", run, data / "desis_2021.csv", out / "paper_numbers.json"])
    compare_values(load_json(out / "paper_numbers.json"), load_json(source / "paper_numbers.json"))
    report["checks"]["principal_paper_numbers"] = "PASS: all saved summary values"
    command("supp_extra", [ROOT / "scripts/supp_extra_numbers.py", run,
                           run / "base/split_manifest.csv", data / "desis_2021.csv", out / "supp_extra.json"])
    compare_values(load_json(out / "supp_extra.json"), load_json(source / "supp_extra.json"))
    report["checks"]["principal_supplement_numbers"] = "PASS: all saved summary values"
    command("components", [ROOT / "scripts/recompute_components.py",
                            run / "base/controlled_ablation.csv", out / "controlled_ablation.json"])
    compare_values(load_json(out / "controlled_ablation.json"), load_json(source / "principal/controlled_ablation.json"))
    report["checks"]["controlled_components"] = "PASS: all eight settings, gate and scale contrasts"

    supplement = out / "supplement"
    command("historical_replay", [ROOT / "supplement/verify_selectors_and_directional.py",
                                  "--data-dir", data, "--out", supplement])
    for name in ("single_split_identity_and_validation", "original_state_replay_and_cells",
                 "R8_selector_search_and_bootstrap_verified", "R5_geographic_records_verified"):
        report["checks"][name] = load_json(supplement / f"{name}.json")
    command("supplement_diagnostics", [ROOT / "supplement/verify_saved_diagnostics.py",
                                      "--replay-dir", supplement])
    report["checks"]["supplement_diagnostics"] = load_json(supplement / "saved_diagnostics_verified.json")
    command("S11", [ROOT / "supplement/recompute_s11.py", "--out", supplement])
    actual = load_json(supplement / "S11_seed20261018.json")["conditions"]
    expected = load_json(ROOT / "supplement/expected/S11_seed20261018.json")["conditions"]
    for condition in expected:
        # CSV text formatting may differ between NumPy versions; compare all numerical fields.
        for values in (actual[condition], expected[condition]):
            values.pop("source_sha256")
    compare_values(actual, expected)
    report["checks"]["S11"] = {"seed": 20261018, "draws": 5000, "all_values_match": True}
    command("S24", [ROOT / "supplement/recompute_s24.py",
                     ROOT / "supplement/records/results/vista_smst_r8_external_2020_20260812_attempt3/R8_PER_LOCATION.csv",
                     out / "S24.json"])
    compare_values(load_json(out / "S24.json"), load_json(ROOT / "supplement/expected/S24_shared_crop_intervals.json"))
    report["checks"]["S24"] = {"seed": 20260812, "draws": 1000, "all_values_match": True}
    report["status"] = "PASS"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: complete verification report: {out / 'verification_report.json'}", flush=True)


if __name__ == "__main__":
    main()
