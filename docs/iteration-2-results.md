# Iteration 2 results — bounded feasibility restoration

Version 0.3.0 · 16 October 2026 UTC.
[Plan](iteration-2-plan.md) · [API](api.md) · [Validation](validation.md).

## Repository review and baseline

Work starts from the public `iteration-1` branch at `e627721` (review of the
first delivery). The public `main` branch still contains the original interface
draft at `d2d2cf1`. A separate local worktree was created from `iteration-1`;
the older local ZIP sources were not used as the development base.

The reviewed commit fixes four important first-delivery issues: the merit
penalty update, scale-aware QP acceptance, bound/domain handling in the derivative
checker and propagation of recovered Python exceptions. It also adds HS071,
iteration history and orbit-transfer fixtures. All of those changes are retained.
Local baseline verification reproduced **57 Rust tests plus one doctest and
67 Python tests** before implementation.

The selected gap is concrete: HS071 from `[1,1,1,1]` has inconsistent
linearized rows at the initial point even though the NLP is feasible. The
baseline stops there. Other candidate work, especially QP reuse and second-order
corrections, is intentionally deferred to keep this iteration measurable.

## Implemented

A private `crates/sqp/src/restoration.rs` constructs bound-constrained,
regularized Gauss–Newton models of the squared row violation. No additional
slack variables are introduced. The existing Clarabel adapter, signed-dual QP
checks, variable-box contract and numerical hardening are reused.

The driver enters restoration only after an infeasible ordinary QP. An accepted
restoration point must decrease the feasibility merit and have a complete,
finite grouped evaluation. It then resets curvature and penalties, clears stale
NLP multipliers/residuals and retries ordinary SQP. Only the unchanged NLP KKT
checks can declare convergence. Objective increases during restoration are
permitted; restoration-QP multipliers are never presented as NLP multipliers.

| Interface change | Meaning |
| --- | --- |
| `max_restoration_steps=10` | Total accepted restoration-step budget; zero reproduces old behavior |
| `restoration_step_limit=1.0` | Absolute infinity-norm direction bound |
| `restoration_regularization=1e-4` | Positive diagonal Gauss–Newton regularization |
| `restoration_failed` | No resolvable descent or no acceptable complete feasibility trial |
| `restoration_limit` | Enabled restoration-step budget exhausted |
| Restoration report fields and `RestorationRecord` | Attempts, accepted steps, inclusive time and optional typed before/after history |

Rust and Python expose the same controls and reports. The default method
remains the dummy fixture; native calls still require `method="sqp"`. Package
versions are 0.3.0 and the locked external dependency versions are unchanged.
CI and the local check script now also run `examples/restore_hs071.py`.

## Measured improvement

The unchanged independent evaluator in `examples/compare_baseline.py` accepts
all **13 of 13** native SQP cases, compared with **12 of 13** at the reviewed
baseline. The same SLSQP comparison passes **12 of 13**. Models, starts,
objective references and acceptance thresholds are unchanged. Extra reporting
fields expose restoration counts and total QP solves.

For HS071 from `[1,1,1,1]`:

| Quantity | Reviewed iteration 1 / restoration disabled | Iteration 2 |
| --- | --- | --- |
| Status | `linearization_infeasible` | `converged` |
| Accepted steps | 0 | 8: one restoration, seven ordinary SQP steps |
| Grouped functions / derivatives | 1 / 1 | 9 / 9 |
| Objective | 4 at the infeasible start | 17.01401723898 |
| Primal violation | 36 | 8.93e-8 |
| Stationarity | unavailable | 4.18e-7 |
| Complementarity | unavailable | 3.58e-8 |
| Dual-sign residual | unavailable | 0 |

The feasibility step decreases `psi` from 936 to 328.5. Its point is not itself
certified feasible or optimal. Ordinary SQP then reaches the accepted solution.
The SLSQP run from this start still reaches its 100-iteration cap near the
infeasible start. This is evidence about this fixed fixture, not a general
ranking of the algorithms.

All other native comparison cases use zero restoration steps. The orbit-transfer
cases at 84 and 164 variables remain accepted. Raw results, including every
external failure and call count, are in
[iteration-2-baseline.json](iteration-2-baseline.json). Single-run wall times
are recorded only for reproducibility; no speedup claim is made.

## Review of the delivered iteration

The delivery built and passed its 65 Rust tests, doctest and 75 Python tests,
and the comparison above. The comparison has only one restoration case, so the
review added a stress test: `examples/stress_random.py` generates problems that
are feasible by construction (‖x − target‖² on [−2, 2]ⁿ, n = 2–7, with 1–3
equality and 0–3 inequality rows of the form `a·x + s‖x − y‖²`) from random
starts, many of which have inconsistent linearizations. The independent
evaluator judges first-order acceptance. Over four seeds (1,200 problems) it
found three defects; each is fixed with a test that fails on the delivered code.

1. **Restoration crept towards infeasible stationary points of ψ.** Where no
   feasible point is reachable locally, Gauss–Newton converges only linearly
   to a minimizer of ψ with ψ > 0. The absolute test `predicted <= 0` let it
   creep (ψ unchanged to six digits for several steps) until the budget of 10
   was spent, so six runs reported `restoration_limit`, suggesting more budget
   would help. Rerun with a budget of 100, the three from seed 1 ended
   `restoration_failed` after 10 to 17 steps. Restoration now stops when a step not cut by the
   direction bound predicts less than `1e-8·ψ`. The two `restoration_limit`
   results left were still reducing ψ (by roughly 1e-5 relative per step);
   given the budget, they stop as stationary at steps 12 and 18.
2. **A step overshooting a nearly active bound stalled the solve.** After a
   restoration step one variable sat 3e-14 below its upper bound. The
   interior-point QP step exceeded that room by ~1e-11, the step-length cap
   `(u − x)/p` cut the whole step to α ≈ 0.006 and then to 0, and the solve
   ended `stalled` at primal violation 0.06 (SLSQP solves the problem). Step
   components that exceed their step bound by at most the feasibility
   tolerance are now moved onto it, generalizing the fixed-variable
   correction, and a trial component that rounding of `x + αp` places a few
   ulps beyond a bound is set to it. About 2% of capped trials overshoot that
   way, and the delivered line search then halved α.
3. **A step at the QP's noise level stopped the solve just short of
   convergence.** With stationarity 1.18e-6 (tolerance 1e-6) and `pᵀBp ≈ 1e-12`,
   the predicted merit decrease (−1.6e-12) was within the QP's complementarity
   noise, and the solve ended `stalled`. This exit came from the iteration-1
   review. Such a step is now taken once at full length if the merit does not
   rise by more than the QP accuracy (at most three in a row); one more step
   converges.

| Four seeds, 1,200 problems | Delivered | Reviewed |
| --- | ---: | ---: |
| Converged, independently accepted | 1184 | 1186 |
| `restoration_failed` | 8 | 12 |
| `restoration_limit` | 6 | 2 |
| `stalled` | 2 | 0 |
| Restoration off: converged | 969 | 970 |
| SciPy SLSQP: first-order accepted | 1165 | 1165 |

No run reported `converged` at a point the evaluator rejects, before or after;
the script now fails if one does. The 13-case comparison is unchanged
(13 / 13 accepted, identical iteration counts). The 14 remaining failures are
restoration stopping at, or still approaching, a stationary point of ψ (two of
them in a corner of the box); SLSQP fails 12 of the same 14. A local
feasibility method cannot do better there.
The delivered test that accepted either restoration status for permanently
inconsistent rows now requires `restoration_failed` after exactly two steps,
at the minimizer of ψ.

## Failure behavior and remaining limits

Tests cover truly inconsistent rows, a stationary infeasible zero-Jacobian
point, strict fixed-variable bounds, restoration disabled, separate outer and
restoration limits, callbacks stopped partway through a trial, observer stop,
cancellation during restoration, undefined functions and derivatives, a fatal
exception after a recovered domain exception, nonfinite output, and overflow
in the Gauss–Newton matrix. No such case becomes a false convergence or nonlinear
infeasibility finding. A partial trial leaves the preceding complete point intact.

The local model stops at an infeasible stationary point of squared violation
(reported as `restoration_failed`); it has no random escape or global
feasibility guarantee. The direction bound `restoration_step_limit` and the
regularization are absolute, so badly scaled variables or rows slow
restoration down. Squared violation is
unscaled and can favor some rows over others; its reduction need not reduce
the maximum row residual on every step. Forming dense `J^T J` can worsen
conditioning. Regularization and finite checks are safeguards, not a solution
to arbitrary scaling or degeneracy.

The 256-variable dense cap, cold QPs, lack of second-order correction and lack
of automatic scaling remain. No actual user trajectory data or 16-core scaling
measurement has been supplied. Further iteration should measure either difficult
restoration starts or QP setup/factorization reuse on the orbit fixture before
expanding algorithm scope.
