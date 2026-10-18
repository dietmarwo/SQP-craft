# Iteration 07: Use contiguous products for full CSR rows

Written before the implementation. Previous retained source: sqp-perf30-06.

## Plan

For valid sorted CSR, a row with exactly n entries covers all n columns. Multiply the contiguous value slice and x directly; retain the existing gather path for sparse rows. Preserve the iterator sum order, including nonfinite arithmetic behavior.

Target measurements: csr-dense kernel; independent A*p checks; dense random Jacobians. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
