# Iteration 03: Reuse already validated bound structure in KKT checks

Written before the implementation. Previous retained source: sqp-perf30-02.

## Plan

Public bound checks continue validating every input. Inside KKT residuals, bounds have just been validated. Share the arithmetic loop through a crate-private entry point with known bound shapes, avoiding two redundant complete bound-validation passes while retaining finite point and overflow checks.

Target measurements: kkt kernel; QP independent acceptance checks. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
