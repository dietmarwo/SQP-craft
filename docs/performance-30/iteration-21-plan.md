# Iteration 21: Skip zero KKT primal direction coefficients

Written before the implementation. Previous retained source: sqp-perf30-20.

## Plan

The factor-based KKT solve accumulates one J column per coefficient, even when that coefficient is exactly zero. Avoid those axpy passes; retain every nonzero/NaN coefficient and leave the independent original-data acceptance check unchanged.

Target measurements: qp-zero/qp-fixed; KKT polishing; sparse-gradient structured cases. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
