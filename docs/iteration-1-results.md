# Iteration 1 implementation and validation

Version 0.2.0 · 16 October 2026 UTC. [Starting commit](baseline.md) and
[archived revision 2 implementation plan](iteration-1-plan.md).

## Delivered and bounded scope

| Work area | Implementation |
| --- | --- |
| Numerical boundaries | Raw bounds checks, finite KKT intermediates and nonfinite QP warm-start validation |
| Derivative tools | Bound-aware coordinate finite differences with multiple step sizes, grouped cache ordering and explicit inconclusive/skipped coordinates |
| QP adapter | Clarabel 0.11.1, two-sided-to-conic conversion, signed multipliers, original QP KKT acceptance and active-loop cancellation |
| Native driver | Shifted step QPs, L1 merit/Armijo backtracking, domain-error retries, consistent-multiplier damped BFGS and bounded Hessian resets |
| Result/control interfaces | Explicit method selection, first-order termination checks, resource budgets, counters, phase timings and bounded QP trace |
| Python boundary | Owned callback references; native detach and per-callback reattach on the calling thread; original exception identity/traceback preserved |
| Acceptance | Rust/Python analytic fixtures, process isolation, derivative faults, undefined trials, strict boxes, callback budgets and independent SLSQP comparison |

The native driver lives in `crates/sqp`, rather than adding nalgebra to
`crates/core`. This preserves the dependency-free core. It is a dense BFGS
research pilot capped at 256 variables, even though Jacobian input is sparse.
Application inputs remain unavailable, so the plan's analytic-pilot restriction
applies. The complete target-card workload and performance gate remain open.

## QP choice and capability evidence

A compiled probe against the pinned HiGHS native QP API solved
`H=[[2,1],[1,2]], g=[-3,-3]` at approximately `[1,1]`, confirming triangular
storage and the nonzero off-diagonal convention. Inspection of its QP active
loop (`HiGHS/highs/qpsolver/quass.cpp`) found iteration/time checks but no
registered user interruption callback. LP/MIP callback support does not imply
QP callback support.

The plan permits Clarabel if a mandatory HiGHS QP capability is missing.
Iteration 1 therefore uses **Clarabel 0.11.1 only** for production step QPs.
The adapter converts lower CSR to upper CSC without doubling; split conic
row duals reverse-map to the checked-in signed convention. Equality and fixed
variables use zero cones; free sides produce no artificial constraint.

The adapter requires SPD Hessians and rejects unsupported QP warm-start
components explicitly. It uses cold starts, reports raw status/iterations and
rejects reduced-accuracy or limited iterates. An integration unit test injects
cancellation from iteration 1 inside the real Clarabel callback and verifies
`CallbackTerminated`; it uses no timing race. There is no active-set or
factorization reuse claim. A second QP backend is not shipped.

## Algorithm and interface decisions

See [api.md](api.md) for formulas, exact defaults and termination semantics.
Convergence checks fresh original-unit NLP KKT residuals. QP success, a small
step or a feasible candidate alone is insufficient. Linearized infeasibility
is reported as `linearization_infeasible`, never NLP infeasibility.

Every callback point passes the strict variable-box contract. A tolerance-sized
step on a fixed variable may be made exactly zero, with a recorded correction
and complete QP recheck. Starts are never clipped. Rejected trials use functions
only; accepted trial derivatives immediately follow functions at the same x.
Undefined functions/derivatives trials consume counts, clear saved recoverable
exceptions and retry at a shorter step. Fatal callback/observer errors remain
errors. Partial trials never replace the last complete accepted evaluation.

`max_callback_calls` counts grouped functions plus derivatives. The cooperative
time budget includes callbacks, QPs and driver work. The QP receives the remaining
budget. A user callback cannot be forcibly interrupted; use process isolation
when that is required. The observer reports the initial point as iteration 0
and completed accepted iterates. Native computations detach from Python;
callbacks remain serial on the solve's calling thread. Rust model callbacks
still require neither Send nor Sync.

## Analytic baseline results

[iteration-1-baseline.json](iteration-1-baseline.json) contains the actual
single-run data produced by `examples/compare_baseline.py`, using SciPy SLSQP
1.18.1 and the same analytic models/starts. Both solvers passed the independent
objective, primal, stationarity, complementarity and dual-sign acceptance
checks in all eight cases (16 total solver runs).

| Model and start | Native SQP accepted steps | SLSQP iterations |
| --- | ---: | ---: |
| Quadratic `[0,0]` | 1 | 2 |
| Quadratic `[-2,4]` | 1 | 2 |
| Equality `[0,0]` | 3 | 4 |
| Equality `[4,-3]` | 1 | 2 |
| Circle `[0.4,0.2]` | 1 | 8 |
| Circle `[0.2,0.4]` | 6 | 7 |
| Active bound `[1]` | 1 | 2 |
| Rosenbrock `[-1.2,1]` | 34 | 36 |

The external residual check reconstructs active multipliers by least squares;
it does not claim they are SLSQP's returned multipliers. Grouped-call counts
include adapter calls made to keep the callback cache protocol valid; compare
like-for-like application evaluations before drawing cost conclusions. Timings
are single runs of tiny cases, with no warmup/performance significance.

## Validation and remaining limits

[validation.md](validation.md) records exact gates and counts. All original 32
Rust integration tests, doctest and 42 Python contract tests remain. The new
fixtures test QP off-diagonals and lower/upper/fixed duals, nonlinear and bound
optima, bad derivatives including omitted Jacobian entries, strict domain boxes,
undefined starts/trials, exception identity, observer stop, budgets, infeasible
linearizations, BFGS damping/reset guards and spawned workers.

Deferred: restoration/elastic mode, second-order corrections, automatic model
scaling, exact/user Hessians, sparse/limited-memory BFGS, QP warm starts, internal
parallelism and distribution. The pilot can fail on Maratos-effect cases,
poorly scaled models or infeasible linearizations. First-order stationarity is
not a global-minimum or second-order certificate. Application/16-core results
remain unmeasured and the target card is unchanged.

## Next measured task

Obtain one bounded representative trajectory model with a validated Jacobian and
reference start. Run the derivative checker first, then compare unchanged
models/starts against SLSQP using original-unit residual acceptance. Enable a
small QP trace and profile model callbacks, QP work and BFGS separately. If the
first blocker is infeasible linearization or line-search stagnation, evaluate
restoration or a second-order correction before expanding the dense size cap.
No extension should be selected from the tiny-fixture timing numbers alone.
