# Iteration 27: Solve fully unconstrained QPs from the checked Cholesky factor

Written before the implementation. Previous retained source: sqp-perf30-26.

## Plan

If the fresh bounded-source list is empty and the QP is nonelastic, its unique minimizer solves H*x=-g. Use two triangular solves plus one original-H refinement, then require ordinary independent QP acceptance and budget/cancellation checks. Rejecting a direct candidate falls back to Goldfarb–Idnani rather than proving anything.

Target measurements: qp-free; analytic coupled SPD free QPs; cancellation/indefinite guards. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
