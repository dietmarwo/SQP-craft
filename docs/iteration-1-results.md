> Historical iteration-1 record. Current results: [iteration-2-results.md](iteration-2-results.md).

# Iteration 1: native SQP pilot

Version 0.2.0 · 16 October 2026. Implements the
[revision 2 plan](iteration-1-plan.md) on top of the interface draft (`main` at
`d2d2cf1`). Validation details: [iteration-1-validation.md](iteration-1-validation.md). Contract:
[api.md](api.md).

## Scope delivered

| Area | Implementation |
| --- | --- |
| Numerical hardening | Checked KKT intermediates (an overflow is an error, never a zero residual), `Bounds::checked_violation`, finite QP warm-start validation |
| Derivative checker | Coordinate finite differences inside the box, several step sizes, every output row compared, fixed variables reported separately |
| QP adapter | Clarabel 0.11.1: two-sided rows to cones, signed multipliers, scale-aware independent acceptance, cancellation inside the QP |
| Native driver | Step QPs in shifted coordinates, Powell per-row L1 merit penalties, Armijo backtracking with undefined-point retries, damped BFGS with SPD checks and bounded resets |
| Results and control | Explicit `method="sqp"`, first-order convergence test, iteration/callback/time budgets, run report, optional QP trace and iteration history |
| Python | Owned callback references, native phases detached from the interpreter, original exceptions preserved, recovered domain errors dropped |
| Acceptance | Analytic fixtures, HS071, a 164-variable orbit-transfer collocation, an independent evaluator and SciPy SLSQP on the same models and starts |

The driver lives in `crates/sqp`, so the core crate stays dependency-free. It
is a dense-BFGS research pilot capped at 256 variables; the Jacobian input is
sparse but `B` is dense and the QP is re-solved cold each iteration.

### QP backend choice

A compiled probe of the HiGHS QP API solved `H = [[2,1],[1,2]], g = [-3,-3]` at
`[1,1]`, confirming its triangular storage. Its QP active-set loop
(`highs/qpsolver/quass.cpp`) checks iteration and time limits but has no user
interruption callback, so the plan's fallback applies: Clarabel is the only
production QP backend in this iteration. There is no active-set or
factorization reuse, and no second backend ships.

## Review of the delivered iteration

The delivered code built and passed its 47 Rust and 61 Python tests. Probing it
beyond its fixtures found four problems, fixed here with tests that fail on the
delivered code:

1. **The merit penalty stalled the line search on the orbit transfer.** The
   plan's nondecreasing rule `ρ = max(ρ, 1.1‖λ‖∞ + 1)` kept the first QP's
   multiplier (ρ ≈ 33) while the multipliers settled near 1.6. With a 20×
   oversized penalty, the curvature-induced constraint violation dominated the
   merit and most steps were cut to α = 1/32: N = 10 took 357 iterations and
   1884 rejected trials, N = 20 and N = 40 hit the 500-iteration limit. The plan
   proposed that rule as "a pilot policy to validate"; it did not validate.
   Powell's per-row update `μ_i ← max(t_i, ½(μ_i + t_i))`, `t_i = 1.1|λ_i|`
   (the SLSQP/NLPQL rule, with a 10% margin) converges from the same initial
   guess in 54 / 86 / 131 iterations with at most one rejected trial.
2. **Correct QP solutions were rejected.** The adapter required the maximum
   QP residual below an absolute `1e-7`. Interior-point complementarity is
   `|multiplier| × gap`; with multipliers near 500 a 1e-9 gap gives 5e-7, so
   `min (x0+1)² + (x1−3)⁴ + 0.1x0x1, x0 ≥ 0` from `(0, −3)` ended in
   `qp_failure` at the first iteration. One scale-aware rule,
   `QpModel::accepts` (dual-side residuals `<= tol·(1 + max(‖g‖, ‖λ‖, ‖z‖))`),
   now serves the adapter and the driver; it still rejects a flipped multiplier
   sign.
3. **The derivative checker failed every model with a fixed variable.** Fixed
   coordinates made the report `inconclusive`, hence not `passed`, although the
   solver never moves them. Trajectory models fix their initial state, so a
   correct orbit model could not pass. Fixed variables are now listed in
   `fixed` and do not affect `passed`. Also: an undefined perturbed point no
   longer aborts the check (and no longer leaks an `UndefinedPointError` to
   Python), and a point just inside a bound uses a full one-sided step instead
   of a step shrunk to the distance to the bound (1e-13 gave a roundoff error of
   ~1e3 on a value of 1e6).
4. **Recovered Python errors depended on a hook in the core trait.** The
   delivery added `NlpProblem::discard_undefined()`, which every solver had to
   call after recovering from an undefined point, or the binding would re-raise
   the stale exception after a successful solve. The derivative checker did
   not call it. The binding now keeps only the exception of the most recent
   callback and re-raises it only when the solver failed with exactly the
   error that stood in for it; the hook is gone from the core trait.

Also changed: an interior-point QP leaves active rows slightly short of their
bounds, so near convergence the predicted merit decrease can turn non-positive
by up to the QP's complementarity residual. After the Hessian-reset retry such
a step now ends `stalled` with the measured QP accuracy in the message, instead
of `no_descent`. A `Solved` QP is kept even if the deadline passed during it.
Statuses have descriptive messages, the API document no longer contradicts
itself on threading, CI runs debug tests (overflow checks on) and the baseline
comparison, and the archive's checksums and local logs were dropped in favour of
CI and this record.

## Comparison with SciPy SLSQP

`examples/compare_baseline.py` runs both solvers on the same models and starts
and decides acceptance with one independent NumPy evaluator: primal violation
`<= 1e-7`, a multiplier certificate with stationarity and complementarity
`<= 1e-6` and sign residual `<= 1e-8`, and the objective against a reference.
For sqp-craft the certificate is its own multipliers; SLSQP returns no bound
multipliers, so its certificate is reconstructed by least squares. Data:
[iteration-1-baseline.json](iteration-1-baseline.json) (SciPy 1.18.1, one run
each; timings are not a performance claim).

| Case | Start | sqp-craft | Iter. | f / ∇f calls | SLSQP | Iter. | f / ∇f calls |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Quadratic | (0, 0) | accepted | 1 | 3 / 2 | accepted | 2 | 3 / 2 |
| Quadratic | (−2, 4) | accepted | 1 | 3 / 2 | accepted | 2 | 3 / 2 |
| Equality | (0, 0) | accepted | 3 | 4 / 4 | accepted | 4 | 4 / 4 |
| Equality | (4, −3) | accepted | 1 | 3 / 2 | accepted | 2 | 3 / 2 |
| Circle | (0.4, 0.2) | accepted | 7 | 8 / 8 | accepted | 8 | 8 / 8 |
| Circle | (0.2, 0.4) | accepted | 5 | 6 / 6 | accepted | 7 | 7 / 7 |
| Active bound | (1) | accepted | 1 | 2 / 2 | accepted | 2 | 2 / 2 |
| Rosenbrock | (−1.2, 1) | accepted | 34 | 54 / 35 | accepted | 36 | 49 / 36 |
| HS071 | (1, 5, 5, 1) | accepted | 5 | 6 / 6 | accepted | 16 | 47 / 12 |
| HS071 | (2, 2, 2, 2) | accepted | 7 | 8 / 8 | accepted¹ | 15 | 12 / 11 |
| HS071 | (1, 1, 1, 1) | `linearization_infeasible` | 0 | 1 / 1 | iteration limit | 100 | 350 / 101 |
| Orbit transfer, N = 20 (n = 84) | crude guess | accepted | 104 | 106 / 105 | accepted | 110 | 112 / 110 |
| Orbit transfer, N = 40 (n = 164) | crude guess | accepted | 167 | 168 / 168 | accepted | 180 | 181 / 180 |

¹ SLSQP reported "Positive directional derivative for linesearch", but the
point passes the independent check.

Both solvers fail HS071 from (1, 1, 1, 1): there the linearized rows require
`Σp ≥ 24` and `Σp = 18`. sqp-craft reports this after one evaluation, as the
plan requires without a restoration phase; SLSQP spends 100 iterations without
leaving the start's neighbourhood. On the orbit transfer, QP solves dominate the
time (3.7 of 4.1 s at N = 40): a cold interior-point solve with a dense
164 × 164 Hessian each iteration.

## Limits

No restoration or elastic mode (an inconsistent linearization ends the solve),
no second-order correction (the Maratos example from Nocedal & Wright 15.4
converges but rejects up to about two trials per iteration), no automatic scaling, no
exact or limited-memory Hessians, no QP warm starts, no internal parallelism.
Convergence is first-order only. The application target card is still open; no
application data were available, so all results are on analytic and textbook
problems.

## Next measured task

Run the actual low-thrust model: derivative check first, then sqp-craft and
SLSQP from the same starts with `record_history=True`, judged by the
independent evaluator in original units. Profile callbacks, QP and BFGS time
separately. Then choose the next change by the observed blocker:

- an inconsistent linearization: elastic mode or a feasibility restoration phase;
- slow tails with many rejected trials: a second-order correction;
- QP time dominating at larger N: a sparse or structured Hessian with QP warm
  starts (the HiGHS active-set comparison planned for milestone 3).
