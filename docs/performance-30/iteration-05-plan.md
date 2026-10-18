# Iteration 05: Accumulate symmetric Hessian rows locally

Written before the implementation. Previous retained source: sqp-perf30-04.

## Plan

The symmetric lower-storage product repeatedly loads/stores out[i] and reloads p[i] for every term. Hold each row sum and p[i] locally while preserving the original addition order; scatter only contributions to earlier rows. This removes redundant memory traffic without changing the product or scalar operation order.

Target measurements: hessian fixed-work kernel; active-set refinement; QP acceptance. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
