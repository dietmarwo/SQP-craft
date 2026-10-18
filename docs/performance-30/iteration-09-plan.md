# Iteration 09: Skip zero multiplier rows after finite-data validation

Written before the implementation. Previous retained source: sqp-perf30-08.

## Plan

The internal KKT arithmetic receives a fully validated finite Jacobian and finite multipliers. A zero row multiplier contributes no transpose product, so skip its entire sparse row there. Keep the public raw sparse transpose operation unchanged so its nonfinite arithmetic behavior is preserved.

Target measurements: kkt fixed-work kernel with zero multipliers; QP independent checking. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
