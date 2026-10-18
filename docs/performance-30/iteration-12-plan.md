# Iteration 12: Stop negligible-step scans once a material component is found

Written before the implementation. Previous retained source: sqp-perf30-11.

## Plan

A BFGS step is negligible only if every component is below 1e-14. Use the equivalent short-circuit predicate for finite steps instead of computing the complete infinity norm. Nonfinite components must not be mistaken for a negligible zero step.

Target measurements: bfgs kernel; negligible and nonfinite update handling. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
