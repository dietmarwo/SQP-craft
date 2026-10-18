# Iteration 08: Specialize singleton CSR row products

Written before the implementation. Previous retained source: sqp-perf30-07.

## Plan

One-entry Jacobian rows are common in restoration and fixed-variable constraints. Replace iterator/index machinery with one gather and multiply, retaining the leading zero addition used by the original sum. Keep dense and general sparse paths.

Target measurements: restoration NLP; qp-zero; single-entry Jacobian acceptance checks. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
