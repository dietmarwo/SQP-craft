# Iteration 11: Separate contiguous BFGS triangle computation from mirroring

Written before the implementation. Previous retained source: sqp-perf30-10.

## Plan

Compute each lower column through contiguous source/output/vector slices, then mirror the completed triangle in a separate pass. This removes strided mirror writes and nalgebra indexing from the arithmetic loop without changing scalar formulas or update commit semantics.

Target measurements: bfgs kernel; exact matrix-expression test; BFGS phase timings. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
