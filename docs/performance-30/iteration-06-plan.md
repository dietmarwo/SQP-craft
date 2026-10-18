# Iteration 06: Use contiguous arithmetic for dense lower Hessian rows

Written before the implementation. Previous retained source: sqp-perf30-05.

## Plan

A validated lower-triangle CSR row of length i+1 contains exactly columns 0..i. Use contiguous coefficient/input/output slices in that case, retaining the sparse gather fallback. Preserve the row-local sum and scatter order from iteration 05.

Target measurements: hessian kernel; dense QP refinement; both NLP backends. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
