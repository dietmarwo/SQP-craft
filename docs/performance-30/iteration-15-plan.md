# Iteration 15: Avoid clearing completely stored Hessian triangles

Written before the implementation. Previous retained source: sqp-perf30-14.

## Plan

After lower-triangle CSR validation, n*(n+1)/2 stored entries prove that every lower entry is supplied. In that case overwrite the complete factor triangle directly and avoid zeroing it first. Sparse/incomplete Hessians keep the safe lower-triangle clear from iteration 14.

Target measurements: active-set conversion; full BFGS NLP matrices; incomplete Hessian regressions. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
