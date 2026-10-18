# Iteration 29: Reuse Cholesky factors only for bit-identical Hessians

Written before the implementation. Previous retained source: sqp-perf30-28.

## Plan

Retain an owned complete Hessian key (shape, CSR structure and every f64 bit). If it exactly matches a previously successful factorization, reuse L and its conditioning estimate; otherwise clear validity, reload and refactor. Failed/invalid factorizations must never become cache hits. Report factor reuse as a QP update, and test changed values/patterns/signed zeros/dimensions/invalid inputs. Add a coupled free-QP fixture to avoid judging iteration 27 only on already-diagonal cases; rebuild checkpoint 28 with the same extended harness first.

Target measurements: repeated qp-dense and qp-free-dense; cache invalidation; setup/update accounting. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
