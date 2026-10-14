# VISTA

Cross-Sensor Data Preparation for Spectral World Modeling.

This package provides VISTA's five frozen logistic-classifier models,
PRISMA-only inference, and the current buffered five-fold common-utility
experiment scripts and reported outputs. CPU execution is sufficient;
no GPU is required. It prepares complete 59-band spectra while preserving
a fixed classifier's annotation of the global affine fallback. This is not
a guarantee of spectral improvement or true crop classification.

## Frozen-model inference

Use Python 3.12 and `requirements-inference.txt` for inference. Inference
needs only NumPy and PRISMA measurements. It never reads DESIS,
crop ground truth, or coordinates. Reflectance is expressed in percentage
points; negative values are retained.

```bash
python predict.py --model models/fold_0.npz --input prisma_common59.npy --output corrected.npz
```

The input is an N-by-59 NumPy array on the wavelengths stored in the model.
The Python `interpolate_prisma` function accepts native PRISMA wavelengths
and performs the original linear interpolation without extrapolation.

The five models are the original spatial cross-validation states, not five
random seeds or a newly fitted all-data production model. Use each model's
declared held-out fold when reproducing the paper. Selecting a model for a
new acquisition requires a separate, justified validation decision.

## Reconstruct a frozen fold

```bash
python train_original.py --desis desis_2021.csv --prisma prisma_2021.csv --fold 0 --output rebuilt_fold_0.npz
```

This reuses the original fold's selected hyperparameters, fits only its
development rows, and does not select from held-out outcomes. It does not
rerun or replace any archived model automatically. Use `requirements-fit.txt`
for the recorded fitting environment. The original source is retained under
`vendor/`; the portable inference implementation is NumPy-only.

## Reproduce the current comparison

Install `requirements-fit.txt`, then stage the public 2021 CSV files from
[USGS, DOI 10.5066/P14CQACU](https://doi.org/10.5066/P14CQACU):

- `desis_2021.csv`: SHA-256 `85ac21e38fb990a75714f90be0d2ced643275adf4e46570f326e765709c48ccc`
- `prisma_2021.csv`: SHA-256 `0c6ef3dff0e5eff1c3e9ebb349494465d4459ee189c2aeb089af4e7622822647`

```powershell
.\REPRODUCE_CURRENT_CU.ps1 -DataDir C:\path\to\public\inputs -Python C:\path\to\python.exe
```

This command fits and evaluates the locked current five-fold protocol,
including full-ridge and PDS controls and the paired-reference decoder
anchor. New outputs go under `reproduced/`; archived `reported_results/`
are not overwritten. The public CSV files are not bundled. DESIS is used
for fitting or scoring after prediction, never as deployment input.
`train_original.py` does not reproduce this complete control comparison.

The archived experimental scripts are unchanged. The inference API rejects
non-finite intermediate results, and the PowerShell wrapper stops on a failed
Python command. These engineering safeguards leave frozen models, formulae,
selection decisions, and verified real-data predictions unchanged.

## Scope and limitations

- Original paired 2021 DESIS–PRISMA task: 16,581 locations, 59 bands and five crops.
- Original signed-hash spatial folds and inclusive 0.02-degree exclusion buffer.
- Exact deterministic top-label agreement; inadmissible candidates return the fallback.
- Declared and severe spectral error events are evaluation diagnostics, not deployment gates.
- The current experiment source includes logistic and random-forest realizations.
  Separate historical OL/2020 estimator binaries are not included; this package
  does not claim to regenerate every historical manuscript result.
- Spectral world-model dynamics are not evaluated by this package.
- Release candidate for review; no journal submission, acceptance, or complete
  historical reproducibility certificate is implied.

No license has been selected. This prepared archive is not a public licensed release.
