# Iteration 02: Check the Hessian triangle through sorted row endpoints

Written before the implementation. Previous retained source: sqp-perf30-01.

## Plan

QpModel::validate already validates strict CSR ordering. A row contains an above-diagonal entry iff its last column exceeds its row index. Use this endpoint on valid input and scan only to construct a precise error for invalid input. Reduce lower-triangle validation from O(nnz) to O(n).

Target measurements: validate kernel; active-set conversion; both QP backends. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
