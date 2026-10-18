# Iteration 04: Share checked KKT arithmetic after QP validation

Written before the implementation. Previous retained source: sqp-perf30-03.

## Plan

The QP residual path validates its entire mutable model, validates the step, computes H*p and A*p, checks derived arithmetic, then revalidates the same matrix/bounds through the public KKT wrapper. Extract crate-private checked arithmetic and explicitly validate multiplier lengths/finiteness before using it. Keep the public KKT entry fully validating and keep the elastic path unchanged.

Target measurements: independent QP acceptance; qp-dense/qp-pairs; NLP QP checking phase. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
