# Iteration 16: Avoid unused row norms and extend working-set coverage

Written before the implementation. Previous retained source: sqp-perf30-15.

## Plan

Free rows can never enter the working set, so their normal norm is irrelevant. Skip that dot product while preserving finite-data validation. Coverage inspection found the frozen micro suite lacks free dense rows and saturated equality bases: add free-row, fixed-variable and dense-normal QPs, rebuild the preceding numerical checkpoint with this same common harness, then compare.

Target measurements: qp-free-rows; qp-fixed; qp-dense-normals; original NLP suite. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
