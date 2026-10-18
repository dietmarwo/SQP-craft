# Iteration 19: Restrict Householder matrix updates to exact nonzero support

Written before the implementation. Previous retained source: sqp-perf30-18.

## Plan

Trailing exact zeros in a transformed normal have zero Householder weight, so their J columns are unchanged. Find the last nonzero term and restrict matrix column scans to that prefix. Compute the norm on the original full tail to preserve its reduction order, and retain the general dense case.

Target measurements: structured qp-pairs/qp-fixed; reflector invariants; orbit engine phase. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
