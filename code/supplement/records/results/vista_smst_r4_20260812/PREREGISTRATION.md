# SMST-R4 pre-registration: semantic-margin spectral transport

Locked: 2026-08-12  
Status: new real paired-instrument experiment; no final-set outcome has been computed.

## Scientific question

Can a sensor-specific spectral transport improve the agreement between paired PRISMA and DESIS spectra while refusing a material-conditioned residual when it would change the frozen material decoder's predicted crop state? This is a deterministic property of a frozen classifier on a paired agricultural dataset, not a population-level safety or calibration certificate.

## Data and scope

Primary data are the USGS **DESIS and PRISMA spectral library of agricultural crops for California's Central Valley in August 2021** (DOI `10.5066/P14CQACU`). The metadata identify 16,581 matched locations where the two surface-reflectance products overlap. The experiment uses only their common visible--near-infrared range and only the five supplied crop labels for evaluation.

Input hashes:

* `desis_2021.csv`: `85AC21E38FB990A75714F90BE0D2CED643275ADF4E46570F326E765709C48CCC`
* `prisma_2021.csv`: `0C6EF3DFF0E5EFF1C3E9EBB349494465D4459EE189C2AEB089AF4E7622822647`
* reconciled metadata XML: `1453CF193438E59AD61D4CDDE2EB4C089C1C28C843517585F7EB47C0F4E5AA46`

This is one co-located agricultural region and one acquisition period. It will not be described as a cross-region, cross-season, population, or universal sensor-adaptation result.

## Fixed representation and partitions

1. Treat DESIS as the reference instrument. Linearly interpolate the PRISMA reflectance values to every DESIS center at or above 407 nm; exclude only DESIS `d403`, which lies outside PRISMA's supplied first center. Reflectances remain in the supplied percentage scale.
2. Form 0.01-degree spatial blocks as `floor((longitude + 180) / .01), floor((latitude + 90) / .01)`. Assign every block deterministically by `SHA-256("smst-r4-20260812:" + block_id) mod 10`: 0--5 development, 6--7 validation, 8--9 final. A run stops before final evaluation if any crop class has fewer than 20 final samples.
3. Fit all standardizers, source material decoder, global transfer, and material-conditioned residual maps on development rows only. Use validation labels only to choose among the explicitly listed regularization/threshold grid. The final split is read only after all choices and the final decision lock are persisted.

## Methods

Let $z$ be the PRISMA spectrum interpolated to the DESIS common bands and $y$ the matched DESIS spectrum.

* **Identity:** $z$.
* **Global ridge transport:** a per-band affine ridge map $T_g(z)$ fitted on development pairs without crop labels.
* **Unconstrained semantic mixture:** a frozen class-balanced $L_2$-regularized softmax material decoder $f$ is trained on development DESIS spectra and crop labels. Its posterior on $T_g(z)$ weights class-indexed residual affine maps. Those residual maps are fitted only using the decoder's development pseudo-class posterior, not the true target crop label. This produces $T_s(z)$.
* **SMST (proposed):** commit $T_s(z)$ only when (i) the top class of $f(T_s(z))$ equals the top class of $f(T_g(z))$ and (ii) its top-two logit margin is at least the margin of $f(T_g(z))$ plus a validation-selected nonnegative margin. Otherwise commit $T_g(z)$.
* **Label-conditioned diagnostic upper bound:** the same residual family weighted by the true crop label. It is an oracle diagnostic only and is never a deployable comparator.

The global ridge penalty is selected from $\{10^{-6},10^{-4},10^{-2},1\}$. Decoder penalty is selected from $\{10^{-5},10^{-3},10^{-1}\}$. Residual ridge penalty is selected from $\{10^{-6},10^{-4},10^{-2},1\}$. The margin grid is $\{0,0.02,0.05,0.10\}$ in decoder-logit units. Validation selection is lexicographic: higher macro-F1, then lower spectral RMSE, then higher SMST coverage, then smaller penalties/margin.

## Final outcomes and comparisons

On the once-opened final split, report accuracy, macro-F1, per-class F1, DESIS-referenced spectral RMSE, SAM (radians), acceptance coverage, and paired deltas of SMST versus global ridge. A declared per-sample harm is the union of a correct-to-incorrect frozen-decoder change relative to global ridge and an RMSE increase greater than both 0.25 percentage points and 5% of the global-ridge RMSE. This is an empirical paired diagnostic, not a risk bound.

The paper may only claim a paired, real-instrument, common-VNIR evaluation if the final lock/audit succeeds. It may not call a result a material-state guarantee beyond the frozen decoder's exact top-label preservation test, or claim generalization beyond this dataset.

## Stop conditions

The final split must not be opened if data parsing, paired-location alignment, split class support, source-fit/validation separation, or decision locking fails. A negative or non-improving final result will be preserved and will not be promoted into the manuscript.
