# Iteration 26: Reuse a fresh equality-source list across attempts

Written before the implementation. Previous retained source: sqp-perf30-25.

## Plan

Derive equality source IDs together with the bounded-source list during model load. Initial working-set construction and a possible cold retry then visit only equalities, in their original order, instead of scanning all source bounds again. Rebuild both lists on every model change.

Target measurements: initial basis setup; qp-fixed/qp-dense; source-list invalidation regression. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
