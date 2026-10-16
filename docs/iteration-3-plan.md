# Iteration 3 plan — measured QP workspace reuse

16 October 2026. Base: reviewed public `main` at
`214f93b0ec0cf407f62fd01ef23df90c7a81fe5c`.

## Review and measured motivation

Iteration 2 is now merged with fixes for restoration stationarity, near-bound
step roundoff and steps at QP numerical accuracy. Its random stress suite is
part of CI. Preserve these changes, all acceptance tolerances and the bounded
local restoration algorithm. The previous delivered ZIP is not the base.

Local baseline: 70 Rust tests plus one doctest and 76 compiled Python tests
pass. Three installed-wheel runs of each orbit case show QP time accounts for
about 87% of end-to-end time at 84 variables and 92% at 164 variables. The
164-variable case makes 168 QP calls and 1,446 interior-point iterations.
Median wall time is about 2.18 seconds on this host; this is a fixture profile,
not an application or hardware-scaling result.

The current adapter constructs a new Clarabel solver for every QP. The pinned
Clarabel 0.11.1 supports numeric data updates with unchanged sparsity/cones.
Its `solve` resets the primal/dual iterate, while retaining allocations,
equilibration and symbolic linear-system structure. This is the selected scope.
Phase timing will distinguish conversion, setup/update, optimization and checks
so a speed claim does not rest on a changed number of outer iterations alone.

## Public interface and compatibility

- Add `SqpOptions.qp_reuse_workspace: bool = false` in Rust and Python, and
  `QpOptions.reuse_workspace: bool = false` at the backend seam. It is opt-in.
- Add per-QP setup/update/retry counters and conversion/setup/update/solve/check
  timings to `QpReport`; aggregate them with `qp_` prefixes in `RunReport`.
  Existing `qp_solves` counts backend calls, while `qp_iterations` includes all
  engine attempts, including a retry. Existing inclusive `qp_seconds` stays.
- Keep defaults, original-unit NLP KKT tests, independent QP acceptance,
  cold-start rejection of supplied QP warm starts, and solver statuses intact.
  Workspace reuse is not a primal/dual warm start, numeric-factor reuse or a
  promise of identical iteration trajectories.
- Bump package versions to 0.4.0. No external dependency updates.

## Cache and failure design

Keep at most one workspace per `ClarabelQp` instance; no global/shared cache.
The key must compare dimensions, both CSC index patterns, ordered cone kinds
and the signed mapping back to original row/variable multipliers. Equality,
finite-side, sparsity or dimension changes force setup. Numeric values,
gradient and right-hand sides must all be replaced on every cache hit. A
restoration model normally changes structure and therefore rebuilds.

Use the existing Clarabel defaults and its `is_data_update_allowed()` guard.
If presolve changed structure, do not cache that solver. Do not disable
presolve or change scaling merely to obtain reuse. Clarabel's pinned source is
the authoritative behavior: the older public guide is more restrictive about
presolve than the implementation's structural-change check.

Replace settings and the cancellation/deadline callback on every engine call.
Drop any partially updated workspace on an update error. A reused solve that
does not produce an independently accepted optimum gets at most one fresh
setup retry, except cancellation and time/iteration exhaustion, which stop
immediately. Retry uses remaining time and remaining QP iterations, and its
cost is fully reported. Do not retain a workspace after an unsuccessful solve.
Cloning a backend copies configuration/reporting but starts with an empty cache.

Every accepted QP result still passes `QpModel::accepts` in original coordinates.
Reusing initial equilibration may affect conditioning; retry mitigates numerical
failures but cannot promise identical results. Keep this path opt-in.

## Work packages and gates

1. Reproduce the reviewed baseline and freeze its orbit profile.
2. Refactor the adapter into conversion, workspace setup/update and execution;
   add phase telemetry before judging the improvement.
3. Implement exact structural matching, complete numeric updates, invalidation,
   replacement of callbacks/settings, and bounded cold retry.
4. Wire controls and aggregated reports through normal and restoration QPs,
   Rust exports, PyO3 and Python types without changing convergence rules.
5. Add QP tests for changed P/q/A/b, structural zeros, cones/sides, dimension
   changes, original dual signs, failed update/acceptance fallback, cancellation,
   time/iteration limits, cloning and disabling reuse.
6. Add Python tests for opt-in behavior, new diagnostics, reuse on the orbit
   fixture, restoration transitions and preserved callback/error contracts.
7. Compare enabled/disabled paths on the existing 13-case evaluator and the
   same four seeds of 300 random problems. Every `converged` result must pass
   the existing independent evaluator. Publish all failures and any changed
   success counts, without weakening thresholds.
8. Add a repeated, alternating-order benchmark on the 84/164-variable orbit
   cases. Record residuals, objective, outer/inner iterations, call counts,
   setup/update counts and phase times. Report medians and actual outcomes;
   do not assert timing thresholds in CI or generalize to 16 cores.
9. Run debug Rust tests, formatting, Clippy, Rust 1.85, rustdoc, clean native
   wheel Python tests/type checking and dummy-only build checks. Package full
   source, plan, results and reproducible benchmark commands.

## Deferred

No second-order correction, automatic scaling, warm starts, sparse Hessian,
new factorizer, larger variable cap, parallel SQP or global restoration escape.
No actual low-thrust target data or SNOPT comparison is available. The next
algorithmic choice should follow remaining failure profiles or application data.

## Completion

Implemented as version 0.4.0. The acceptance and implementation gates above
pass: 79 Rust tests plus one doctest, 85 compiled Python tests and 42
dummy-only contract tests. Both workspace modes accept the existing 13 cases
and 1,186 of 1,200 random problems, with no false convergence. Five repeated
orbit measurements show median wall-time reductions of 9.9% and 15.6% at
84 and 164 variables on this host. Reuse remains opt-in. Full implementation
details, raw data and remaining limits accompany this plan in
`docs/iteration-3-results.md` and `docs/validation.md` in the source ZIP.

## Sources

- [Clarabel data-update guide](https://clarabel.org/stable/user_guide_data_updating/).
- Pinned [data-updating implementation](https://github.com/oxfordcontrol/Clarabel.rs/blob/v0.11.1/src/solver/implementations/default/data_updating.rs):
  complete numeric updates and structural guards.
- Pinned [solve loop](https://github.com/oxfordcontrol/Clarabel.rs/blob/v0.11.1/src/solver/core/solver.rs):
  `default_start` on each call and checked settings replacement.
- Existing `docs/iteration-2-results.md`, `examples/orbit_transfer.py` and
  `examples/stress_random.py` supply the local baseline and acceptance fixtures.
