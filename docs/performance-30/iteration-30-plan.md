# Iteration 30: Cache the unrotated inverse only after confirmed Hessian reuse

Written before the implementation. Previous retained source: sqp-perf30-29.

## Plan

The initial J=L^-T and its norm depend only on the checked Hessian, not gradient/rows/bounds or the warm set. Cache them only after a cross-call factor reuse confirms that H repeats, avoiding an extra matrix copy on nonrepeating NLP Hessians. Reuse copies the unrotated factor before working-set transformations; every Hessian miss invalidates it. Test changed Jacobians/bounds and H invalidation.

Target measurements: repeated qp-dense/qp-dense-normals; cache promotion/invalidation; original NLP work. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
