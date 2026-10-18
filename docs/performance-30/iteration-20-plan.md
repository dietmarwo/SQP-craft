# Iteration 20: Build exact diagonal inverse factors directly

Written before the implementation. Previous retained source: sqp-perf30-19.

## Plan

When every strictly lower Cholesky entry is exactly zero, L is diagonal and L^-T contains only reciprocal diagonal entries. Detect this exact structure and fill J directly instead of performing all zero dot-product triangular solves. Any nonzero entry, however tiny, keeps the general solve.

Target measurements: diagonal qp-free/qp-box/qp-pairs; initial and restoration QPs; tiny off-diagonal regression. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
