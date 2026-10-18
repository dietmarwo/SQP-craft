# Iteration 10: Fuse finite BFGS validation with triangle construction

Written before the implementation. Previous retained source: sqp-perf30-09.

## Plan

Every BFGS entry is computed once in one triangle and mirrored. Track finiteness while computing that triangle rather than scanning both full triangles afterwards. Keep all scalar rank-two formulas and the subsequent block positive-definiteness check unchanged; a failed update must leave B unchanged.

Target measurements: bfgs fixed-work kernel; native BFGS phase; exact-expression and rejection tests. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
