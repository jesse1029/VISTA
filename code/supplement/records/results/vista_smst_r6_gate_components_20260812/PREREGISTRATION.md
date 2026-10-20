# SMST-R6: fixed gate-component characterization

Locked: 2026-08-12  
Status: post-R4 secondary analysis; it cannot replace, retune, or upgrade R4.

## Purpose

R6 isolates the two clauses of the already locked SMST admission rule on the
already opened R4 final split.  It answers a narrower descriptive question:
how do the top-label and nondecreasing-margin clauses separately change the
use of the *same* residual candidate?  This is neither a new primary test nor
an additional validation search.

## Frozen inputs

R6 reads the immutable R4 paired input tables, the R4 development model state,
and R4's final decision lock.  It requires the exact R4 runner digest
`3A14E29F063F5809D941F4B202483BB017345475164C6C10A6EBA90CFFD770F1`, the
R4 preregistration digest
`7618C6C5859CF43E84A7A158121C02872BB519C8D548331012D1C1C6A18AE04A`, and the
locked configuration `(lambda_g, lambda_f, lambda_r, delta) =
(10^-2, 10^-1, 10^-4, 0)`.  No parameter, model, split, or threshold is
searched or selected in R6.

## Four fixed candidates

For every R4 final location, let `G` be global ridge, `S` the unconstrained
semantic residual, `L` the test that their frozen top labels agree, and `M`
the test that the frozen top-two margin of `S` is at least that of `G`.
R6 reports exactly these four fixed decisions:

1. `unconstrained`: always use `S`;
2. `label_only`: use `S` iff `L`;
3. `margin_only`: use `S` iff `M`;
4. `both`: use `S` iff `L and M`, which must reproduce R4 SMST exactly.

For each decision R6 reports coverage, frozen-decoder label changes relative to
`G`, accuracy, macro-F1, paired RMSE and SAM deltas relative to `G`, and the
same declared spectral-harm predicate used by R4.  The predicate is a paired
diagnostic, not a safety certificate.  The labels occur only in final metrics;
they never select a residual or gate branch.

## Interpretation boundary

The expected exact equality of the `both` row and R4 is an integrity condition,
not a replication.  These component rows remain conditional on one
DESIS--PRISMA agricultural release and one frozen decoder.  They cannot
establish cross-region, cross-season, population, causal, or universal claims.
