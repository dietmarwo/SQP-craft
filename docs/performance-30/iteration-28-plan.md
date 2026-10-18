# Iteration 28: Check diagonal box minimizers before general working-set search

Written before the implementation. Previous retained source: sqp-perf30-27.

## Plan

For an exactly diagonal SPD Hessian, clamp -g_j/H_jj to each variable interval and form bound multipliers. This is the exact box-relaxation minimizer; accept it only if the unchanged original full-QP residual test also accepts every row. Otherwise fall back to the general solver. Elastic QPs keep their existing path. Count zero working-set changes for accepted closed-form proposals.

Target measurements: qp-box/qp-zero/qp-fixed; constrained fallback and bound multipliers. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
