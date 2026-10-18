# Iteration 24: Reserve the retained working set once per dimension

Written before the implementation. Previous retained source: sqp-perf30-23.

## Plan

At most n independent normals can enter a basis. Reserve capacity for n members and multipliers during model load, reusing existing allocations across calls. This avoids repeated growth/copying of the first equality/warm working set without changing its contents or ordering.

Target measurements: first QP setup/engine work; saturated and dense-normal QPs. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
