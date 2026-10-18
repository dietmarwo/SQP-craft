# Iteration 22: Compute only the constraint activity when its error scale is unused

Written before the implementation. Previous retained source: sqp-perf30-21.

## Plan

Polishing and full-step length calculations discard the magnitude returned by Sources::value. Add an activity-only sparse dot for these two callers, preserving the same zero-initialized sum. Keep magnitude-aware value computation for violation and feasibility decisions.

Target measurements: qp-dense-normals; active-set polishing/engine phase. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
