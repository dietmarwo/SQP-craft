# Iteration 14: Clear only the factor triangle that is read

Written before the implementation. Previous retained source: sqp-perf30-13.

## Plan

Active-set Cholesky and inverse-transpose routines read only the lower triangle of their column-major factor. Resize its reusable buffer and clear that triangle, leaving irrelevant upper entries untouched. Continue overwriting every supplied Hessian entry and clearing all missing lower entries.

Target measurements: active-set conversion; diagonal/sparse QPs; repeated dimension changes. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
