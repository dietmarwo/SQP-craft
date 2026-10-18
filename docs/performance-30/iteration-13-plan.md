# Iteration 13: Copy exact-curvature differences without damping arithmetic

Written before the implementation. Previous retained source: sqp-perf30-12.

## Plan

When the curvature condition selects an exact BFGS update, Powell theta is exactly one. Copy y into the retained r buffer instead of computing y + 0*Bs component by component. Keep repaired and damped cases unchanged and retain the exact matrix-expression regression.

Target measurements: BFGS phase; fixed-work update; damped and augmented fixtures. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
