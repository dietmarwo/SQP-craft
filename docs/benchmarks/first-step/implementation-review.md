# Implementation review: SQP items 1 and 2

## Commits

The cumulative patch is based on SQP-craft `55195896ab3a0b94afe67bad4c7148dfbe8955bb`; implementation/evidence and synchronized 0.20.0 release preparation are reviewable together. No release tag is created by the patch.

## Verification

See `summary.json`, `tables.md`, the retained baseline/final panel and stress records, and the combined implementation log for exact executed checks and counts. Tests cover callback failures remaining errors, undefined projected starts, non-finite augmented trial merit, complementarity overflow with SOC, malformed QP output, extreme reflector tails, and full/block/partitioned/exact curvature semantics.

## Notes for review

Internal arithmetic has a distinct `Error::Numerical` origin. Callback-output validation keeps `Error::Evaluation`. The driver catches only the internal arithmetic kind. QP output is checked before independent acceptance; existing checks and tolerances are unchanged. A failed projected start clears the candidate. Accepted-point residual fields are cleared before recomputation so an overflow cannot leave residuals associated with a previous point.

The step safeguard is implemented separately from the error fixes. Its default decision must follow the predeclared benchmark gates; an unsuccessful performance experiment still ships as an explicit option, disabled. Jobs freeze both effective safeguard values in their resolved payload and discovery examples/tests account for that resolution.

## Merge order

The status/error fixes logically precede the performance options and their evidence. Review both before releasing 0.20.0. The maintainer creates the tag after the normal release workflow gates; publishing and registry credentials are outside this patch.
