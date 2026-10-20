# SMST-R7: frozen-configuration spatial-fold rotation

Locked: 2026-08-12  
Status: secondary same-release robustness characterization; no model-selection step.

## Question and scope

R7 asks whether the fixed R4 SMST configuration shows the same *direction of
paired spectral change and decoder-relative commitment* across five disjoint
spatial holdouts of the USGS DESIS--PRISMA agricultural release.  It is
explicitly not an independent sensor pair, region, season, or population test.

## Fixed design

Use the three immutable R4 inputs and their recorded hashes, 60 common VNIR
bands, and exactly `(lambda_g, lambda_f, lambda_r, delta) =
(10^-2, 10^-1, 10^-4, 0)`.  No validation grid is run.  Build a 0.01-degree
block identifier as in R4 and assign each block to one of five folds by
`SHA-256("smst-r7-spatial-5fold-20260812:" + block_id) mod 5`.  For each fold,
fit all maps and the source DESIS decoder on the other four folds, then evaluate
the held-out fold once.  The run fails rather than omits a fold if a held-out
fold has fewer than 10 instances of a released crop class.

## Outcomes and interpretation

For all five folds, report test count, class support, SMST coverage, residual
candidate label changes, post-gate label changes, macro-F1 delta, RMSE delta,
SAM delta, and R4's fixed spectral-harm diagnostic.  Report every fold, not a
selected subset.  The exercise is a post-R4 robustness description: it does
not relock R4, provide independent validation, establish safety, or support a
claim about unobserved sensor pairs.
