# SMST-R5: post-R4 regional stress test

Locked: 2026-08-12

## Status and purpose

This is a secondary, post-R4 stress characterization on the same USGS
DESIS--PRISMA release. It is not an independent dataset, not a blind test
relative to R4, and will not be promoted to the primary result. It asks a
narrower question: when the R4 configuration is reused without retuning, does
SMST retain its decoder-relative commitment property across a disjoint
west/east spatial separation within this release?

## Immutable inputs

The data and their hashes are exactly those of R4:

* `desis_2021.csv`: `85AC21E38FB990A75714F90BE0D2CED643275ADF4E46570F326E765709C48CCC`
* `prisma_2021.csv`: `0C6EF3DFF0E5EFF1C3E9EBB349494465D4459EE189C2AEB089AF4E7622822647`
* reconciled metadata XML: `1453CF193438E59AD61D4CDDE2EB4C089C1C28C843517585F7EB47C0F4E5AA46`

PRISMA is interpolated to the 60 DESIS bands from 410--1000 nm exactly as in
R4. No raw imagery leaves the mounted remote data volume.

## Fixed regional split and models

Let `t = -120.2425085` degrees longitude, the release median longitude fixed
from metadata before this R5 run. A 0.01-degree exclusion band avoids adjacent
training/test rows:

* **West:** longitude `<= t - 0.01` (7,642 rows).
* **East:** longitude `>= t + 0.01` (7,765 rows).
* The intervening rows are excluded from both arms.

Run precisely two predeclared directions: west-trained/east-evaluated and
east-trained/west-evaluated. Fit the decoder, global transport, and residual
maps only on the direction's training side. Reuse R4's locked configuration
without a validation search: global ridge `1e-2`, decoder `1e-1`, residual
ridge `1e-4`, and margin `0`.

## Reported outcomes and boundaries

For each direction report source-decoder accuracy/macro-F1, paired RMSE/SAM,
SMST coverage, unconstrained semantic label changes relative to global ridge,
SMST label changes relative to global ridge, and the R4 declared spectral-harm
count. Target crop labels evaluate only metrics; residual weighting and the
gate consume frozen decoder outputs.

The paper may describe this only as a same-release, geographically separated
stress test. It must not call it independent cross-region validation,
cross-season evidence, a second sensor-pair evaluation, or a population-risk
certificate. A negative result remains in the evidence record and will not be
removed or retuned.
