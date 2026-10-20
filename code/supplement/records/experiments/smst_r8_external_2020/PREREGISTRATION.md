# R8: frozen-2021 to external-2020 DESIS--PRISMA comparison

## Question

Does the 2021-locked SMST bundle retain any paired-spectral benefit on the
separate USGS 2020 DESIS--PRISMA release, and is that behavior distinguishable
from standard calibration-transfer maps and non-margin selectors?

## Evidence boundary

The 2020 release is a temporally separate paired acquisition and a distinct
USGS data release (DOI 10.5066/P98LO5D4). It uses the same sensor pair and a
California agricultural setting. It is therefore an external temporal
evaluation, not a new sensor-pair or geographic-population replication.

## Frozen sources and partitions

* The R4 2021 script, development set, validation set, selected bundle, and
  final partition are immutable authorities. R8 must not read the R4 final
  labels, paired targets, or R4 final result records when selecting any R8
  baseline or selector.
* The R8 external data comprise only the June 2020 PRISMA observation and the
  matching June 2020 DESIS observation, paired by Unique ID. The August DESIS
  observation is excluded before any calculation because it has no same-date
  PRISMA counterpart in this release.
* Both external spectra are linearly interpolated onto the fixed 60-band R4
  DESIS grid only when that grid lies inside each source wavelength range.

## Comparators selected on R4 development and validation only

1. R4 diagonal affine ridge.
2. Full multivariate ridge.
3. PLS regression.
4. Piecewise direct standardization (PDS), using local input windows.
5. An unconditional diagonal residual on top of the selected diagonal map.
6. Hard k-means-conditioned diagonal residual maps, where clusters are fitted
   without crop labels.

Each method uses the finite grids implemented in the accompanying script and
is selected by lower 2021 validation paired RMSE, then by the declared simpler
configuration order. The R4 SMST bundle is not refitted.

## Selector comparison for the fixed R4 residual candidate

The following rules are evaluated: always residual, top-label-only,
raw-margin-only, top-label plus raw margin (SMST), and top-label plus
temperature-scaled posterior confidence, entropy, or normalized logit margin.
Temperature and any matched-coverage cutpoint are fitted only on R4 validation
rows. Each ranked rule retains the same “accepted / top-label-compatible”
fraction as the raw SMST conjunction on R4 validation, applied to the number
of top-label-compatible rows in the external set and rounded deterministically.
This preserves a frozen conditional operating rule under shift; external
unconditional coverage is reported rather than forced to be identical.

## Endpoints

The primary external endpoints are paired RMSE and SAM versus the 2020 DESIS
reference. We also record frozen-decoder labels, observed-label macro-F1,
commitment coverage, label transitions versus the diagonal fallback, declared
RMSE harms (the existing 0.25 and 5 percent diagnostic), and the corresponding
conditional-on-commit harm rate. Spatial-cell bootstrap intervals use the
fixed 0.01-degree cell definition and 1,000 deterministic resamples.

The run additionally writes per-wavelength bias and RMSE, complete
per-location accounting, parameter counts, and fitted/evaluation times.
No endpoint is a population-risk certificate, a universal safety claim, or a
claim of independent sensor-pair generalization.

## Stop rules

The run stops rather than adapts if paired IDs, metadata, source hashes,
wavelength support, or required R4 authority hashes do not match. No grid is
expanded, no crop subset is selected after inspection, and no 2020 outcome
changes a 2021 configuration.
