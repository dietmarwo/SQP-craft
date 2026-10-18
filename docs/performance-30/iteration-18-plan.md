# Iteration 18: Skip dependent insertions after the basis is full

Written before the implementation. Previous retained source: sqp-perf30-17.

## Plan

Once q==n there is no free transformed component and every further normal is dependent. Return before projecting additional equality/warm-start candidates. Preserve insertion order before saturation and let the normal independent feasibility checks handle inconsistent dependent rows.

Target measurements: qp-fixed saturated basis; dependent-equality and infeasibility tests. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
