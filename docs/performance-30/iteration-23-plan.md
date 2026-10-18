# Iteration 23: Borrow the original gradient during nonelastic polishing

Written before the implementation. Previous retained source: sqp-perf30-22.

## Plan

Only elastic rows modify the QP gradient. In ordinary polishing borrow model.gradient directly instead of copying it into ge on every settle/refinement. Elastic polishing continues rebuilding its gradient from the current model and elastic terms.

Target measurements: ordinary active-set engine phase; qp-dense/qp-pairs; elastic regression suite. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
