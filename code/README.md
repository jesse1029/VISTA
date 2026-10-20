# VISTA

Annotation-consistent cross-sensor spectral harmonization for world-model inputs.

VISTA maps PRISMA spectra to the DESIS reference scale. A class-weighted residual correction
refines a global affine harmonization, a label-consistency gate returns the corrected spectrum
only when a frozen classifier assigns it the same label as the global harmonization, and a
validation-selected correction scale controls the correction strength. Every returned spectrum
keeps the class assigned to the global harmonization by that same classifier.
CPU execution is sufficient; no GPU is required.

## Verify the reported results

Use Python 3.12 in a separate environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-verify.txt
python prepare_inputs.py --out inputs
python verify_reproduction.py --data-dir inputs --out verified
```

This command runs inference with the saved models and recomputes numerical summaries.
It checks every held-out VISTA spectrum in the five-fold experiment, the saved spectra
and summary values for the comparison methods, the single-split 2021 model and validation
records, frozen-model transfer to 2020, geographic-test summaries, and the two declared
supplementary bootstrap calculations. It performs no training or parameter selection.
The program exits with an error when a required file, input hash, row alignment, or
numerical comparison fails. Choose a fresh output directory for each run.

`verified/verification_report.json` records the executed checks and their results;
individual stage logs are in the same directory. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for the mapping to the paper and the distinction between model replay and retraining.
The bundled input tables come from the public USGS releases described in
[data/README.md](data/README.md), with the exact research-input hashes in `input_hashes.json`.

## Frozen-model inference

Inference needs only NumPy and PRISMA measurements; it never reads DESIS, crop labels, or
coordinates. Reflectance is in percentage points; negative values are retained.

```bash
pip install -r requirements-inference.txt
python predict.py --model models/fold_0.npz --input prisma_common59.npy --output corrected.npz
```

The input is an N-by-59 array on the wavelengths stored in the model; `FrozenVISTA.interpolate_prisma`
converts native PRISMA wavelengths by linear interpolation without extrapolation. The five models
are the spatial cross-validation states of the principal comparison, one per outer fold. Applied
to their held-out PRISMA spectra, they reproduce the reported VISTA outputs to within 1e-13.

To rebuild one frozen model from the public tables instead of using the stored file, run

```bash
python train_original.py --desis desis_2021.csv --prisma prisma_2021.csv --fold 0 --output fold_0.npz
```

It refits the global harmonization, the logistic classifier, and the residual map on that fold's
development partition with the parameters in `selected_folds.json`, without reselection.

## Reproduce the principal comparison

Download the public 2021 tables from USGS ScienceBase (DOI 10.5066/P14CQACU):

- `desis_2021.csv`: SHA-256 `85ac21e38fb990a75714f90be0d2ced643275adf4e46570f326e765709c48ccc`
- `prisma_2021.csv`: SHA-256 `0c6ef3dff0e5eff1c3e9ebb349494465d4459ee189c2aeb089af4e7622822647`

```bash
pip install -r requirements-fit.txt
bash reproduce.sh /path/to/inputs          # Linux or macOS
# .\REPRODUCE_CURRENT_CU.ps1 -DataDir C:\path\to\inputs   (Windows)
```

This fits and selects every configuration of the buffered five-fold comparison by the stated
validation rule (no locked decisions): ridge regression, PDS, PLS, kernel ridge regression, and an
MLP, each without and with the label-consistency gate, and VISTA with logistic and random-forest
classifiers. It then runs the fixed-map correction-scale analysis, the reference-classifier
agreement, and the main and supplementary summaries. The runner writes `base/split_manifest.csv`;
`paper_numbers.json` and `supp_extra.json` contain the corresponding numerical summaries.
Outputs go to `reproduced/`; the archived
`reported_results/` are not overwritten. The Linux script launches four comparison workers;
the PowerShell script runs sequentially. Elapsed time depends on the CPU and numerical libraries.

| Script | Purpose |
|---|---|
| `scripts/run_vista_comparison.py` | Principal comparison (archived protocol, converged classifier) |
| `scripts/run_additional_harmonizers.py` | PLS, kernel ridge, MLP, and rule-selected ungated PDS |
| `scripts/crossed_scale_analysis.py` | Fixed residual maps at unit and selected scale |
| `scripts/controlled_ablation.py` | Fixed residual maps with and without the gate (component table) |
| `scripts/reference_anchor.py` | Classifier agreement on the paired DESIS spectra |
| `scripts/paper_numbers.py`, `scripts/supp_extra_numbers.py` | Summary quantities of the principal comparison |
| `scripts/vista_converged.py` | Loads the archived module with the converged classifier |

### Classifier convergence

The principal results use logistic classifiers fitted on Linux with Newton–Cholesky at tolerance
1e-10. `scripts/vista_converged.py` treats a convergence warning as an error and preserves the
archived classifier's remaining settings, class order, and development-only standardization.
`reported_results/converged_rerun/` contains the current manuscript results. The earlier
L-BFGS run at tolerance 1e-4 is retained in `reported_results/archived_lbfgs_windows/`.
The stored NumPy model states support inference independently of the fitting solver.

The archived Windows L-BFGS execution reported 297 large-error events and 95.09%
adoption for logistic VISTA; the principal Linux execution reports 294 and 95.20%.
The archived ungated PDS also used fixed windows, whereas the principal comparison
uses validation-selected windows. These records identify different executions;
the current entry points implement the principal protocol described above.

The historical scripts `scripts/run_common_utility_expanded.py` and
`scripts/run_current_reference_anchor.py` use the archived L-BFGS protocol. The current
reproduction entry points use `scripts/run_vista_comparison.py` and `scripts/reference_anchor.py`.

## Scope

- Paired 2021 DESIS-PRISMA release: 16,581 locations, 59 bands (410-1000 nm), five crops.
- Signed-hash spatial folds with an inclusive 0.02-degree exclusion buffer.
- Large-error events and union events are evaluation diagnostics, not deployment gates.
- The single-split 2021 study and the 2021-to-2020 transfer use separate saved models and
  source programs under `supplement/`. `verify_reproduction.py` includes their replay
  and result checks.
- Spectral world-model dynamics are not evaluated by this package.

## License

The VISTA code is distributed under the [MIT License](LICENSE).
Public USGS input data retain their original public-domain terms; see
[data/README.md](data/README.md). Dependencies retain their respective licenses.
