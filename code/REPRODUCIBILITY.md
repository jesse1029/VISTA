# Reproducing the reported results

The release provides three distinct operations:

1. `predict.py` applies a saved five-fold VISTA model to PRISMA spectra.
2. `verify_reproduction.py` replays saved models and recalculates reported results.
3. `reproduce.sh` / `REPRODUCE_CURRENT_CU.ps1` fit and select models for the principal
   five-fold comparison. `train_original.py` refits one VISTA fold with its saved
   selected parameters.

The release verification executes operation 2. Training scripts are included for
independent reruns; they were not executed as part of packaging.

## Paper-to-code map

Section names refer to the main paper and supplement, rather than historical table numbers.

| Paper content | Inputs and program | Verification performed |
|---|---|---|
| Principal buffered five-fold comparison; Supplement S5 | `models/fold_*.npz`, `reported_results/converged_rerun/principal`, `additional` | Replay all 16,581 VISTA spectra; check fold roles, adoption, labels and RMSE; recompute all quantities produced by `scripts/paper_numbers.py` |
| Controlled components; main Table II | `principal/controlled_ablation.csv`, `scripts/recompute_components.py` | Recompute all eight settings and the reported gate and scale contrasts from saved rows |
| Complete spectra, wavelength errors, ranges and smoothness; S6 | Ten saved spectral NPZ files; `scripts/paper_numbers.py`, `scripts/supp_extra_numbers.py` | Compare saved spectra against per-location RMSE and recompute numerical summaries, including development-spectrum ranges |
| Single-split partition and selection; S2–S3 | `supplement/records/results/vista_smst_r4_20260812` | Regenerate 9,873 development, 2,824 validation and 3,884 final memberships; replay 48 stored states at 192 recorded operating points and check the recorded selection |
| Single-split 2021 results; S7 | Same saved selected state and original final CSV | Replay all 3,884 final spectra; compare classifications, margins, adoption and RMSE with the original output; verify all five saved output summaries, including the diagnostic oracle |
| Single-split gate components; S8 | Replayed single-split spectra and `supplement/verify_saved_diagnostics.py` | Recompute all four gate variants and their classification, adoption and spectral quantities |
| Wavelength diagnostics | Replayed 2021 and 2020 spectra and `supplement/verify_saved_diagnostics.py` | Check all 59 bands for 2021 global/returned spectra and accepted corrections, and 2020 returned VISTA spectra |
| Directional geographic stress; S9 | `supplement/records/results/vista_smst_r5_regional_20260812` | Verify 7,642 west and 7,765 east memberships; recompute saved directional RMSE, accuracy, macro-F1, adoption and large-error counts |
| Fixed 2021-to-2020 transfer; S10 | Saved 2021 state, public 2020 inputs and original R8 output | Replay 1,490 spectra; verify all seven selector memberships, temperature selection, 778-row matched selections and the 1,000-draw calculation with seed 20260812 |
| Raw-margin stratification; S11 | Replayed single-split and 2020 outputs; `supplement/recompute_s11.py` | Recompute both conditions, 5,000 draws each, with the manuscript seed 20261018 |
| Shared-crop 2020 analysis; S10 | `supplement/recompute_s24.py` | Recompute the four shared crops and aggregate results over 98 cells, 1,000 draws, seed 20260812 |
| Reference-spectrum classifier agreement; S5 and S12 | Saved five-fold and single-split classifiers and paired DESIS spectra | Recompute overall and per-crop agreement for the principal study; overall agreement and the three groups of changed predictions for the single-split study |

The original source scripts and saved outputs are included under
`supplement/records`. Their original directory names (R4, R5, R6, R8) identify study
records, not the current manuscript's table numbering.

## Saved states and full reruns

The five principal models are portable, pickle-free NumPy archives. The selected
single-split model is also supplied as `supplement/single_split_selected_state.npz`.
To audit all 48 historical validation states, the verification command reads the
original model archive only after checking its recorded SHA-256, using a restricted
reader that accepts the expected NumPy arrays and stored estimator fields.

The directional models and fitted 2020 comparison-method states were not serialized
by the original runners. Their original source, selections and available outputs
are included. The verification report distinguishes recomputation from saved rows
from inference using stored models. It does not claim to have rerun those fits.
The 2020 wavelength file's `diagonal_affine` columns belong to the separately selected
comparison baseline, so the wavelength replay checks its `smst` columns only.

The original 2020 runner is
`supplement/records/experiments/smst_r8_external_2020/run_smst_r8_external_2020.py`.
The single-split runner is
`supplement/records/results/vista_smst_r4_20260812/run_smst_r4.py`.
Both expose their inputs through `--help`; the original files remain unchanged so
their recorded source hashes can be checked. The integrated verification entry
provides portable paths for replay and summary calculation.

## Numerical conventions

- Reflectance uses percentage points; the 59 wavelengths span 410–1000 nm.
- Reproduction checks use the original input bytes and compare floating-point
  quantities with an absolute tolerance of `1e-9` (individual VISTA spectra use
  `1e-10`). Integer counts, selected rows and labels must match exactly.
- The principal study and historical single-split study have different split
  definitions and saved states. The verifier checks each against its own records.
- The 2020 PRISMA CSV has a repeated `X951` header. Its second occurrence receives
  the suffix `.1`, matching the original table reader; only digit-only wavelength
  headers enter interpolation.
- The 2020 all-location calculation uses 238 occupied cells. Its label-compatible
  subset uses 216 cells. The shared-crop calculation uses 98 cells.
- The current S11 calculation explicitly uses seed 20261018. Its historical
  5,000-draw seed was not preserved; the integrated command reproduces the current
  manuscript calculation.

## Release checks

`verification/verification_report.json` records the actual packaging-time replay.
The aggregate `paper_numbers.json` now includes `crossed.map_contrasts`, copied
from the existing `principal/crossed_scale.json`; all pre-existing numerical
values were preserved.

`MANIFEST.json` and `SHA256SUMS.txt` cover the distributed files. The unified verifier
checks the complete manifest and input hashes before computation. It rejects optimized
Python execution (`-O`, `-OO`, or `PYTHONOPTIMIZE`) so archived scientific assertions
remain active. Generated outputs go to a separate directory and never replace the
reported results. `python tests/test_entrypoints.py` exercises missing/corrupted-file
rejection, output protection, optimization rejection, inference from an unrelated
working directory, batching, class preservation, invalid inputs and interpolation.
