# Iteration 25: Scan only bounded constraint sources for violations

Written before the implementation. Previous retained source: sqp-perf30-24.

## Plan

Build a retained ascending list of sources with a finite lower or upper bound during model load. Violation selection then skips free variables/rows without repeatedly loading their bounds. Rebuild the list on every model load and test bound-kind changes on the same backend.

Target measurements: qp-free-rows; repeated violation scans; changing-bound invalidation. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
