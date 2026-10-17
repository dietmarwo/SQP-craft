# Contract: problem, callbacks, solver, results

Version 0.4.0. Interfaces are pre-release; `publish = false` prevents
accidental releases.

## The problem

```text
minimize    f(x)
subject to  constraint_lower <= c(x) <= constraint_upper     (m rows, m >= 0)
            lower            <=  x   <= upper                (n variables, n >= 1)
```

There is no hidden sign change of the objective. Equal finite bounds express
equalities; `-inf` lower or `+inf` upper bounds express one-sided or free
entries. NaN, `lower = +inf`, `upper = -inf` and `lower > upper` are rejected.
All numbers are IEEE-754 `f64` in original, unscaled units. No automatic
scaling is applied.

## Problem specification

`ProblemSpec` holds the bounds and the fixed sparsity pattern of the constraint
Jacobian `J[i][j] = dc_i/dx_j`, in zero-based CSR: `jacobian_indptr` has `m + 1`
entries starting at 0 and ending at `nnz`; each row's column indices are
strictly increasing and below `n`. The pattern must contain every entry that
can be nonzero anywhere in the domain, because it defines the order of the
Jacobian values the model returns. An `m = 0` problem has `indptr = [0]` and no
indices.

The specification is validated once, at construction (`ProblemSpec::new` in
Rust, the `ProblemSpec(...)` constructor in Python), and is immutable
afterwards. Errors name the offending field and entry, for example
`variable bounds: entry 1 has lower=5, upper=4`. One spec can serve any number
of solves; in Python it is picklable for process-based workers. A `scipy.sparse`
CSR matrix's `indptr` and `indices` arrays can be passed directly once its
indices are sorted (`A.sort_indices()`).

## Model callbacks

| Rust (`NlpProblem`) | Python model method | Computes |
| --- | --- | --- |
| `functions(x, constraints_out) -> f` | `functions(x) -> (f, c)` | objective and constraint values |
| `derivatives(x, gradient_out, jacobian_out)` | `derivatives(x) -> (g, jac_values)` | dense gradient (length `n`) and Jacobian values in pattern order (length `nnz`) |

The evaluations are grouped the way SQP consumes them: `f` and `c` together at
every line-search trial point, `grad f` and `J` together at every accepted
point. A trajectory model typically gets `f` and `c` from one propagation.

**Guarantees from every solver and tool in this repository**

- `x` has length `n`, is finite and lies within the variable bounds. The model
  is never evaluated outside the box, so it may rely on, say, a positive mass.
  A trial component that rounding of `x + αp` would place a few ulps beyond a
  bound is set to that bound; any other trial outside the box is shortened.
  Points are never projected.
- `derivatives` is only called at the `x` of the immediately preceding
  `functions` call, so intermediate results may be reused. The native solver
  calls `derivatives` only at accepted points, never at rejected trials.
- Rust output buffers have exactly the required length; they are pre-filled with
  NaN, so an entry the model forgets to write is reported as non-finite.
- Python callbacks receive a fresh `list[float]`; mutating it has no effect on
  the solver.

**Model obligations**

- Return finite values with the right lengths. Python callbacks may return
  lists, tuples or 1-D NumPy arrays; the two results must be returned as a
  `tuple`.
- Signal "not defined at this point" (a failed propagation, say) with
  `Error::UndefinedPoint` / `raise sqp_craft.UndefinedPointError`. At a
  line-search trial the solver halves the step and continues; at the start
  point it aborts the solve.
- Any other error aborts the solve.

The callback trait requires neither `Send` nor `Sync`: a model instance is
mutable and local to one solve. Parallel strategies run independent solves with
independent model instances, in separate processes when the model is Python.

## Solving

Rust: `SqpSolver::solve(&mut self, spec, problem, x0, options, cancel, observer)`,
implemented by `DummySolver` (crate `sqp-craft`) and `NativeSqp<B>` (crate
`sqp-craft-solver`) with a QP backend `B`: `ClarabelQp` (crate
`sqp-craft-clarabel`) or `ActiveSetQp` (crate `sqp-craft-activeset`).

Python: `solve(spec, model, x0, *, method="dummy", method_options=None,
options=None, cancellation=None, observer=None)`. `method="sqp"` selects the
native solver, configured by `SqpOptions`; `method_options` is rejected for the
dummy. A method that was not compiled (a wheel built with
`--no-default-features`) raises `NotImplementedError`; there is no fallback.

The start point `x0` must be finite and inside the variable bounds; it is
rejected, never projected or repaired. It may violate constraint rows. Input
errors are reported before any model call and take precedence over a pending
cancellation.

`Options` holds settings shared by all methods: `feasibility_tolerance`
(absolute, unscaled; default `1e-7`; the primal threshold of the convergence
test) and `threads` (only `1`; anything else is `Unsupported` /
`NotImplementedError`).

### `SqpOptions`

Every field is honoured; invalid values raise `InvalidInput` / `ValueError`.

| Field | Default | Meaning |
| --- | --- | --- |
| `max_iterations` | `100` | Maximum accepted steps |
| `max_backtracks` | `25` | Step halvings after the first trial |
| `max_callback_calls` | `2000` | `functions` + `derivatives` calls, including rejected and undefined trials |
| `time_limit_seconds` | `None` | Cooperative budget for the whole solve; the QP receives the remainder |
| `stationarity_tolerance` | `1e-6` | Absolute, `‖∇f − Jᵀλ − z‖∞` |
| `complementarity_tolerance` | `1e-6` | Absolute |
| `dual_sign_tolerance` | `1e-8` | Multipliers whose sign selects an infinite bound |
| `dense_size_limit` | `256` | Variables allowed, 1 to 256 |
| `initial_hessian_scale` | `1` | `B = scale·I` initially and after a reset |
| `max_hessian_resets` | `3` | Total resets (failed BFGS updates and non-descent retries) |
| `qp_solver` | `"clarabel"` | Python only: `"clarabel"` (interior point) or `"active-set"` (dense dual active set, see below); in Rust, the type `B` of `NativeSqp<B>` |
| `qp_tolerance` | `1e-10` | The QP backend's native tolerance: Clarabel's feasibility and gap tolerances, the active-set violation and sign thresholds |
| `qp_max_iterations` | `200` | Per QP, shared across an attempt and its cold retry. Clarabel: interior-point iterations. Active set: drops and re-entries of constraints (first entries are not charged) |
| `qp_reuse_workspace` | `false` | Reuse compatible Clarabel workspaces; see below. No effect on the active-set backend |
| `qp_warm_start` | `true` | Pass the last step QP's working set to the next QP; only backends that report one (the active-set backend) receive it |
| `elastic_mode` | `false` | Solve every step QP as an ℓ1-elastic QP (rows may be violated at a cost), so restoration is not needed; needs a backend with elastic support (`qp_solver="active-set"`), otherwise `Unsupported` / `NotImplementedError` before any model call. See [elastic-mode.md](elastic-mode.md) |
| `elastic_weight` | `1e4` | Initial elastic weight relative to `max(1, ‖∇f(x₀)‖∞)`; multiplied by 10, at most six times, at a stationary point of the ℓ1 penalty with violated rows |
| `record_qp_limit` | `0` | Step QPs kept in `report.qp_trace`, at most 16 |
| `record_history` | `false` | Normal SQP records in `history`, accepted feasibility steps in `restoration_history` |

## The native SQP method

A dense-BFGS line-search SQP for at most 256 variables. Each iteration at `x_k`:

1. **Step QP.** `min ½pᵀBp + gᵀp` subject to
   `lower_c − c(x_k) <= J p <= upper_c − c(x_k)` and
   `lower − x_k <= p <= upper − x_k` (bounds shifted to step coordinates),
   solved by the QP backend and accepted only through `QpModel::accepts`
   (below). A step component that exceeds its step bound by at most the
   feasibility tolerance is moved onto the bound and counted
   (`fixed_step_corrections` for fixed variables, `bound_step_corrections`
   otherwise) before that check. Interior-point steps satisfy bounds only to
   their tolerance, and next to an active bound the remaining room can be far
   smaller (3e-14 in the stress case that exposed this); without the move, the
   step-length cap `(u − x)/p` cut the whole step to almost zero.
2. **Convergence test** at `x_k` with the QP's fresh multipliers: `converged`
   only if primal violation, stationarity, complementarity and dual-sign
   residuals are all within their tolerances. The stationarity and
   complementarity residuals must clear their tolerances by their own
   rounding level, about `(k_j + 2) ε (|∇f_j| + Σ_i |J_ij λ_i| + |z_j|)` for
   entry `j` of `∇f − Jᵀλ − z` (`k_j` rows use variable `j`) and
   `|μ| ε (|v| + |b|)` for a product `|μ|·|v − b|`. Otherwise huge
   multipliers make the residual rounding noise: an interior-point QP on a
   complementarity problem produced multipliers of 5·10⁹ and a stationarity
   residual of 9.5·10⁻⁷ that an independent NumPy evaluation put at
   1.2·10⁻⁶. A small step, a small objective change or QP success alone never
   ends the solve.
3. **Merit line search.** Per-row penalty weights follow Powell's rule
   `μ_i ← max(t_i, ½(μ_i + t_i))` with `t_i = 1.1·|λ_i|` (`μ_i = t_i` at the
   first QP), so a large early multiplier does not keep the penalty large for
   the rest of the run. The merit is `φ(x) = f(x) + Σ μ_i·violation_i(c(x))`
   over finite row sides; variable bounds hold at every iterate. The predicted
   decrease is `−gᵀp + Σ μ_i (violation_i(c) − violation_i(c + Jp))`, which is
   at least `pᵀBp > 0` for an exact QP solution. Trials `x_k + αp`, `α = 1,
   ½, ¼, …` (capped so the box holds), are accepted when
   `φ(x_k + αp) <= φ(x_k) − 10⁻⁴·α·predicted`.
4. **Update.** `derivatives` at the accepted point, then damped BFGS (Powell
   1978) with `y = ∇ₓL(x_{k+1}, λ) − ∇ₓL(x_k, λ)`, the new multipliers at both
   ends. Damping keeps `B` positive definite in exact arithmetic; rounding in
   the explicit update can still leave it indefinite. The QP backend checks
   every `B` it receives exactly (the Clarabel adapter by a separate Cholesky
   factorization, the active-set backend by the factorization it solves with)
   and returns
   `NotPositiveDefinite`; the driver then resets `B` to `scale·I` and re-solves
   that QP, as after a failed update (non-finite values or non-positive
   curvature after damping). Both consume the Hessian reset budget;
   `RunReport.qp_not_positive_definite` counts the rejected QPs. Without
   budget the solve stops as `numerical_failure`.

If the predicted decrease is not positive, `B` is reset once and the QP
retried. If it is still not positive but its magnitude is within the QP's own
accuracy (interior-point solutions stop slightly short of active bounds, which
costs up to the QP complementarity residual per entry), the step is at the QP's
noise level. Near a solution it is still the right step (`pᵀBp ≈ 1e-12` while
stationarity is just above tolerance), so it is tried once at full length and
accepted if the merit does not rise by more than that accuracy; at most three
such steps in a row. Otherwise the solve ends `stalled` (noise level) or
`no_descent`. The active-set backend puts active constraints on their bounds
to rounding, so its steps have no such interior-point gap.

With `qp_warm_start` (the default) the working set of each accepted step QP
is passed to the next one as `QpWarmStart.active_set`, if the backend
reported one. Clarabel reports none, so its runs do not depend on the
setting.

Bounded local feasibility restoration is described below. Elastic mode
(`elastic_mode`, [elastic-mode.md](elastic-mode.md)) replaces it: every step
QP is ℓ1-elastic, rows it leaves violated get the penalty weight γ in the
merit, Powell's rule restarts when the linearization becomes consistent, and
a stationary point of the ℓ1 penalty with violated rows raises γ tenfold, up
to six times, before the solve ends `locally_infeasible`. There is no
second-order correction (so the
Maratos effect can cost backtracking steps), no automatic scaling, and no
exact-Hessian option. First-order convergence is not a global or second-order
optimality certificate.

### Statuses

| Status | Meaning |
| --- | --- |
| `converged` | All four first-order residuals within tolerance at the returned point |
| `iteration_limit`, `evaluation_limit`, `time_limit` | Budget exhausted; the last complete point is returned |
| `cancelled` | Token cancelled or observer stop; the last complete point is returned |
| `linearization_infeasible` | Inconsistent step QP with restoration disabled; no nonlinear infeasibility claim |
| `restoration_failed` | Restoration reached a stationary point of the squared violation, or found no decreasing complete step |
| `restoration_limit` | Restoration step budget exhausted while it was still making progress |
| `locally_infeasible` | Elastic mode: a stationary point of `f + γ·violation` with violated rows, at the largest γ; no nonlinear infeasibility claim |
| `line_search_failure` | No acceptable step within `max_backtracks` (including all trials undefined) |
| `no_descent` | The step is not a merit descent direction, even after a Hessian reset |
| `stalled` | The step is below the QP's accuracy or cannot change the iterate |
| `qp_failure` | QP limit, backend error, or a solution that failed the independent check |
| `numerical_failure` | BFGS failed after the reset budget was spent |
| `stub` | Dummy method only |

## Results

| Field | Meaning |
| --- | --- |
| `status`, `message` | How the solve ended and why |
| `candidate` | The returned point, or `None` if no complete evaluation happened |
| `evaluation` | `f`, `c`, `∇f`, Jacobian values, `primal_violation` and `feasible_candidate` at the candidate |
| `multipliers` | Most recent QP multipliers |
| `stationarity`, `complementarity`, `dual_sign` | Residuals at the candidate with `multipliers` |
| `counts` | `functions` and `derivatives` calls, including rejected and undefined trials |
| `iterations` | Accepted steps |
| `report` | `RunReport` diagnostics |

When the solve stops after a step (a limit, say), `multipliers` come from the
QP at the previous iterate. The residuals computed with them are still
meaningful: any multiplier vector gives an upper bound on the best first-order
residual at the point. Unavailable residuals are `None`, never zero.

`primal_violation` is the largest bound violation over variables and
constraints, absolute and in original units. `feasible_candidate` compares it
with the tolerance. Neither is evidence of optimality, and an infeasible
candidate does not show that the problem is infeasible.

`RunReport` holds the method, QP backend and its last raw status, QP solves and
iterations, rejected and undefined trials, Hessian resets, the elastic-mode
counters (`elastic_steps`: accepted steps whose QP left rows violated;
`elastic_weight`: the final γ, `None` without elastic mode;
`elastic_weight_increases`), fixed-step and
bound-step corrections, the last accepted step norm, inclusive phase times (total,
callbacks including Python conversion, QP, BFGS; not a pure GIL-wait measure),
the optional QP trace, and the optional history. Each `IterationRecord`
describes one iterate where a QP was solved: objective, primal violation,
stationarity and complementarity (the convergence-test values), `‖p‖∞`, the
accepted step length (`None` if no step was taken, as at the final iterate of a
converged run), trials evaluated, the largest penalty weight, and QP iterations.

### The dummy method

1. Validate options and `x0`.
2. Check cancellation; call `functions(x0)` once; check outputs.
3. Check cancellation; call `derivatives(x0)` once; check outputs.
4. Check cancellation; report iteration 0 to the observer; check cancellation.
5. Return `x0` unchanged, `status = stub`, `iterations = 0`, and
   `multipliers`, `stationarity`, `complementarity` all `None`.

## Multiplier sign convention and KKT residuals

At a first-order point

```text
grad f(x) = J(x)^T λ + z
```

where `λ` (one per constraint row) and `z` (one per variable) are the
multipliers; the Lagrangian is `f − λᵀc(x) − zᵀx`. A positive multiplier belongs
to an active lower bound, a negative one to an active upper bound, zero to an
inactive one. Equalities accept either sign. This is the HiGHS convention; the
QP seam uses the same one, so QP multipliers are directly the next NLP
estimates.

`kkt::residuals` / `kkt::nlp_residuals` (Python: `kkt_residuals`) recompute,
independently of any solver and in unscaled units:

| Residual | Definition |
| --- | --- |
| `primal` | Largest bound violation of variables and rows |
| `stationarity` | `‖grad f − Jᵀλ − z‖∞` |
| `complementarity` | Largest `|μ| · |distance to the bound selected by sign(μ)|` |
| `dual_sign` | Largest `|μ|` whose sign selects an infinite bound |

Every intermediate product and difference is checked: an overflow is an
`Evaluation` error, never a residual of zero. `Bounds::checked_violation` is the
fallible form of `Bounds::violation`.

## QP subproblem seam (`sqp_craft::qp`, Rust only)

`QpModel` is the convex QP in the step `p`:

```text
minimize    0.5 pᵀHp + gᵀp
subject to  row_lower <= A p <= row_upper,   variable_lower <= p <= variable_upper
```

`H` is stored as its lower triangle in CSR; an off-diagonal entry stands for
both symmetric positions and is not doubled. Validation checks structure, not
positive semidefiniteness.

A `QpBackend` is configured per call with `QpOptions` (native tolerance,
acceptance tolerance, iteration cap, remaining time, `reuse_workspace` and
`elastic_weight`) and returns a `QpOutcome`:
`Optimal(solution)`, `Infeasible`, `Unbounded`, `LimitReached(last iterate)`,
`Cancelled` or `NotPositiveDefinite` (the Hessian failed the backend's
convexity check and nothing was solved); `report()` gives the backend name,
raw status, limit kind and iterations. Infeasibility, unboundedness and a
non-positive-definite Hessian are outcomes, not errors. A
`QpSolution` carries the step, multipliers in the shared convention, an optional
working set and the iteration count. `QpWarmStart` may carry a step,
multipliers and/or a working set; all are validated (finite, right lengths).

**Acceptance.** `QpModel::accepts(solution, tolerance)` recomputes the KKT
residuals from the model data alone and requires the primal residual
`<= tolerance` (step units) and the stationarity, complementarity and dual-sign
residuals `<= tolerance·scale` with `scale = 1 + max(‖g‖∞, ‖λ‖∞, ‖z‖∞)`. The
scaling keeps correct interior-point solutions with large multipliers (whose
complementarity is `|μ|·gap`) while catching sign, transposition and doubling
errors, whose residuals are of the order of `scale`. The adapter and the driver
use the same rule; the driver's tolerance is
`min(feasibility, stationarity, complementarity tolerance)`.

**Elastic QPs.** With `QpOptions.elastic_weight = Some(γ)` the backend
solves `min ½pᵀHp + gᵀp + γ Σ_i dist(a_iᵀp, [l_i, u_i])` with hard variable
bounds instead, which is feasible whenever the variable bounds are. Row
multipliers then satisfy `|λ_i| ≤ γ`, and `λ_i = ±γ` on violated rows. Only
backends whose `supports_elastic()` is true accept it (the active-set
backend); the others return `Unsupported` from `configure`.
`QpModel::accepts_elastic` (and `..._with_workspace`) applies the same
thresholds to `kkt::elastic_residuals_with_workspace`: the residuals with
each violated row's bounds relaxed to its value, plus
`violation_i·|γ − σ_iλ_i|` in complementarity and `max(|λ_i| − γ, 0)` in
dual sign. Rows violated by at most the primal tolerance count as satisfied
and appear in the primal residual.

`DummyQp` validates its input and returns `Unsupported`; it never fabricates a
step.

### Clarabel adapter (`sqp-craft-clarabel`)

Clarabel 0.11.1, an interior-point conic solver in Rust. It does not solve
elastic QPs (`configure` returns `Unsupported`). Cold starts only:
supplied warm-start data is validated, then rejected as `Unsupported` rather
than ignored. One thread, positive-definite Hessians, at most 256 variables.
Every model's Hessian is checked exactly before conversion: a strictly
diagonally dominant one is accepted by an upward-rounded Gershgorin test, a
pattern that splits into disconnected components is factorized per component,
and any other by a dense Cholesky factorization. A failure is returned as
`QpOutcome::NotPositiveDefinite`, keeping the cached workspace and
conversion data, so the caller's re-solve reuses them. Every backend that
needs a positive-definite Hessian must do the same (see `QpBackend::solve`):
the driver does not factorize `B` itself. Each finite side of a row or variable bound becomes one
nonnegative-cone row (`a·p ≤ u` as `(a, u)`, `a·p ≥ l` as `(−a, −l)`); equal
bounds become a zero-cone row; free sides produce no row. Clarabel's duals
`z ≥ 0` satisfy `Hp + g + Aᵀz = 0`, so `λ = z_lower − z_upper`, and `−z` for an
equality. A Clarabel iteration callback polls the cancellation token and the
deadline. A `Solved` result is kept even if the deadline passed during the
solve; reduced-accuracy (`AlmostSolved`) and limited iterates are never used as
steps. HiGHS was probed first (see the results document); it lacks a user
interruption callback in its QP solver.

### QP workspace reuse

Set `SqpOptions.qp_reuse_workspace=True` in Python (`true` in Rust), or
`QpOptions.reuse_workspace=true` at the backend seam. Both default to false.
This retains one compatible workspace per `ClarabelQp` instance. Python
`solve` creates a fresh backend for each NLP solve; Rust callers may reuse
their backend across solves. A cloned backend starts with an empty cache.
There is no shared global cache.

Compatibility requires identical dimensions (including original row count),
P/A CSC index patterns, ordered cone types and signed original-row/variable
mapping. Numeric P, q, A and b are replaced completely on every hit, including
explicit structural zeros. Equality, finite-side, sparsity and dimension
changes rebuild the workspace. The ordinary/restoration transition usually
changes that key. A solver whose presolve changed its structure is not cached;
the adapter keeps Clarabel's existing presolve and equilibration defaults.

Reuse retains allocations, symbolic structure and the initial equilibration.
Clarabel still resets its primal/dual starting point and recomputes numerical
factors; supplied warm starts remain unsupported. Retained equilibration can
change conditioning and iteration trajectories, so enabling reuse does not
promise identical results or faster execution on every problem.

A failed numeric update discards the possibly partially updated workspace.
A reused result that fails the independent QP check, or returns an unsuccessful
status including infeasibility, gets at most one fresh setup/solve. Thus a reused
infeasibility status alone cannot trigger restoration. Cancellation and
time/iteration limits stop immediately. Both attempts share the per-call QP
iteration budget and remaining cooperative time budget, including conversion,
setup and updates. Clarabel's own time-limit clock is the sum of its root
timers, and Clarabel resets only the solve timer per call. The adapter resets
the setup and post-processing timers before every engine call, so a reused
workspace is not charged for earlier calls. Settings and the
cancellation/deadline callback are replaced before every engine call, and no
callback is retained after it. Unsuccessful
solves are not cached. An accepted result retains the existing solved-at-deadline
semantics described above.

`QpReport` exposes per-call diagnostics; `RunReport` aggregates ordinary and
restoration calls under these names (the Python report has the same fields):

| `RunReport` field | Meaning |
| --- | --- |
| `qp_setups` | Fresh engine setup attempts, including cold retries |
| `qp_updates` | Successful complete numeric updates; a later stop can precede the engine solve |
| `qp_cold_retries` | Fresh retry setups actually started after a failed update or reused result |
| `qp_retry_reason` | Most recent reason requesting a cold retry, even if the remaining budget prevented it; otherwise `None` |
| `qp_conversion_seconds` | Matrix/cone conversion and positive-definiteness check |
| `qp_setup_seconds` | Fresh engine construction |
| `qp_update_seconds` | Numeric updates, including failed updates |
| `qp_solve_seconds` | Engine solve time across all attempts |
| `qp_check_seconds` | Backend solution extraction and independent acceptance checking |

For the active-set backend, conversion is the dense copy of `H` and the
row norms, setup is the Cholesky factorization (one setup per call), solve is
`L⁻ᵀ` and the active-set iterations, and updates are always 0.

Per-QP names omit the `qp_` prefix (`setups`, `updates`, and so on).
`qp_solves` still counts backend calls; `qp_iterations` and the final
`QpSolution.iterations` include both engine attempts. `qp_status` is the most
recent native status; `qp_retry_reason` preserves why a retry was requested.
The new phase timers are disjoint subsets of inclusive `qp_seconds`; input
validation, structural matching and other overhead explain their difference.
Dummy reports have zero counters/times and no retry reason. Timing is diagnostic,
not a deterministic API guarantee.

### Active-set backend (`sqp-craft-activeset`)

`ActiveSetQp` is the dual active-set method of Goldfarb and Idnani on a dense
Cholesky factor `H = L Lᵀ`; [active-set-qp.md](active-set-qp.md) has the
design, measurements and validation. It needs no dependency beyond the core
crate.

- Every finite bound side is a constraint `σ aᵀp ≥ σ b` (`σ = 1` lower,
  `σ = −1` upper); equal bounds are an equality, which is never dropped.
  Multipliers are mapped to the shared convention (`λ = σ u`), and the final
  working set is returned as `QpSolution.active_set`: `AtLower` or `AtUpper`
  per active side (either for an equality), `Inactive` otherwise. An
  equality that depends on the others may stay `Inactive`.
- A failed Cholesky pivot is `NotPositiveDefinite`. Inconsistent constraints
  are `Infeasible`, claimed only from a point recomputed from the factors.
  `Unbounded` does not occur. Iteration and time limits are
  `LimitReached(None)` with `limit_reason` `"iterations"` or `"time"`; the
  token is polled at every working-set change.
- Warm starts use `QpWarmStart.active_set`: the equalities and the given
  sides start in the working set (dependent ones are skipped), and negative
  multipliers are dropped before the iterations begin, so any working set is
  a valid start. `step` and `multipliers` are validated and ignored. A
  warm-started attempt that ends infeasible or fails the independent check
  is repeated once from a cold start within the same budget
  (`cold_retries`, `retry_reason`).
- `QpReport.iterations` counts the working-set changes after the initial
  working set. `max_iterations` caps the drops and re-entries only, because
  a cold start needs one entry per active inequality.
- Accepted solutions pass `QpModel::accepts` with the call's acceptance
  tolerance; a cold solution that fails it is `Error::Backend` (the driver's
  `qp_failure`).
- Three `n × n` buffers are kept and reused; `reuse_workspace` has no
  effect, and results do not depend on earlier calls. The backend has no
  variable cap of its own.
- Elastic QPs are supported (`supports_elastic()` is true): row multipliers
  are boxed (`[0, γ]`, `[−γ, γ]` for equalities); a row whose multiplier
  reaches `±γ` becomes elastic, with its multiplier fixed in the linear term,
  and re-enters from the other side once it is strictly satisfied. Elastic
  rows are reported `Inactive` in the working set.

## Errors

| Rust `Error` | Python exception | When |
| --- | --- | --- |
| `InvalidInput` | `ValueError` | Malformed spec, start point or options |
| `Evaluation` | `sqp_craft.EvaluationError` (a `RuntimeError`) | Non-finite or wrongly sized model output, arithmetic overflow |
| `UndefinedPoint` | `sqp_craft.UndefinedPointError` (an `EvaluationError`) | Model not defined at the start point |
| `Unsupported` | `NotImplementedError` | Valid but unimplemented request |
| `Backend` | `RuntimeError` | Backend failure outside the driver's status handling |
| — | `TypeError` | Wrongly typed callback results (not a 2-tuple, text, `None`, ...) |

Exceptions raised by Python callbacks or the observer that end a solve are
re-raised unchanged, with their original identity and traceback. An
`UndefinedPointError` the solver recovered from (a line-search trial, a
derivative-check perturbation) is dropped; the binding keeps only the exception
of the most recent callback and re-raises it only if the solver failed with
exactly the error that stood in for it. Errors never carry partial results.

## Cancellation and observers

`CancellationToken` (Rust) / `Cancellation` (Python) is a shared flag checked
before every model call, between phases and inside the QP backend. It cannot
interrupt a running callback. A callback may cancel its own token; another
Python thread can cancel while the solver runs native code, because native
phases release the interpreter. Python signals (Ctrl-C) are checked before each
callback and end the solve with the original `KeyboardInterrupt`. Use process
isolation where a long callback must be interruptible.

The observer receives `Progress(iteration, objective, primal_violation)`:
iteration 0 is the start point, then one event per accepted step. In Rust it
returns `Control::Continue` or `Control::Stop` (closures implement `Observer`).
In Python it returns `True` to stop, `None` or `False` to continue, so
`observer=print` works. A stop ends the solve with `cancelled` and keeps the
last complete point.

## Threading and Python

The core crate has no dependencies. The binding holds owned references to the
Python callbacks, releases the interpreter for native work, and reattaches on
the calling thread for each grouped callback and observer call. No Python
object is used from another thread. Data crosses the boundary by copy; this is
a correctness-first binding, not a zero-copy array API. The callback time in
the report includes conversion and reattachment; measure it on the real model
before deciding on NumPy buffers, batched evaluation or process workers.

## Derivative checker

`derivative_check::check_derivatives` (Python: `check_derivatives(spec, model,
x, *, steps=(1e-4, 1e-5, 1e-6), absolute_tolerance=1e-5,
relative_tolerance=1e-4, max_directions=256)`) compares the supplied gradient
and Jacobian with finite differences of `functions` along each coordinate:

- It calls `derivatives` once, right after `functions` at `x`; perturbed points
  call only `functions`, all within the variable bounds.
- Central differences where both sides have room for the step `h = step·max(1,
  |x_j|)`; otherwise a one-sided step of full size towards the roomier side, so
  a point just inside a bound does not force a tiny, roundoff-dominated step.
- Every objective and row output is compared, so derivatives missing from the
  Jacobian pattern are found (reported with `analytic = 0`).
- The estimate closest to the supplied value over the step sizes is kept; an
  entry fails if `|analytic − estimate| > absolute + relative·max(|analytic|,
  |estimate|)`.
- Fixed variables are listed in `fixed` and do not affect `passed`: they cannot
  be perturbed and the solver never moves them. Coordinates beyond
  `max_directions`, without room inside the bounds, or undefined at every
  perturbation are listed in `skipped` and make the report `inconclusive`.
- An undefined perturbed point is not used; an undefined base point or any
  other model error aborts the check.

`passed` requires at least one checked coordinate, no issues and nothing
skipped. Checker calls are counted separately from any solve.


## Iteration 2: feasibility restoration

When an ordinary SQP subproblem is infeasible, the driver attempts a local
feasibility step before retrying it. This path does not change the ordinary
merit-penalty update, QP acceptance rule or NLP convergence tolerances.

| Additional `SqpOptions` field | Default | Meaning |
| --- | --- | --- |
| `max_restoration_steps` | `10` | Total accepted restoration-step budget; zero disables restoration |
| `restoration_step_limit` | `1.0` | Positive finite infinity-norm direction bound in variable units |
| `restoration_regularization` | `1e-4` | Positive finite diagonal regularization of the Gauss–Newton Hessian |

Let `r_i=c_i-proj_[lower_i,upper_i](c_i)` and `psi=0.5*sum(r_i^2)`.
The restoration QP uses `g=J_active^T r` and
`H=J_active^T J_active + restoration_regularization*I`, with variable step
bounds intersected with the direction bound. All equality rows and currently
violated inequality rows contribute; inactive inequalities contribute zero.
The original inconsistent linear rows are omitted from this QP. There are no
extra slack variables, so the 256-variable cap is unchanged.

An independently accepted QP direction (after the same step-bound moves as an
ordinary step) is backtracked until it stays within the strict variable box and
strictly decreases `psi`, with Armijo constant `1e-4` and prediction `-g^T p`.

Restoration stops with `restoration_failed` when the direction predicts a
decrease of at most `1e-8·psi` without being cut by the direction bound: a
stationary point of `psi` that is not feasible. Gauss–Newton approaches such a
point only linearly, so an absolute test would creep until the step budget ran
out. Near a feasible point the ratio is of order one. The threshold is small
because the absolute regularization scales the ratio down to about
`2σ²/restoration_regularization` on rows whose Jacobian singular values `σ` are
small. A run can still end `restoration_limit` while `psi` is decreasing
slowly towards a stationary value; the restoration history shows the trend. It may increase the objective or the maximum
individual row violation while decreasing the squared total violation.
Undefined function/derivative trials backtrack; other exceptions remain fatal.
Only a complete evaluation can be committed. Arithmetic overflow is an error.

After an accepted feasibility step, B is reset to its initial diagonal,
penalties are cleared and NLP multipliers/residuals become absent. Restoration
QP duals are never published as NLP multipliers. The next ordinary QP supplies
fresh estimates, and only the unchanged original-unit NLP KKT checks can
report convergence. These restarts are counted by `restoration_steps`, separately
from the failed-BFGS `hessian_resets` budget.

All accepted steps, including restoration, count toward `max_iterations` and
observer iteration numbers. Shared callback/time/cancellation budgets apply.
`restoration_failed` means a stationary point of `psi`, no resolvable
feasibility descent or no acceptable complete trial; `restoration_limit` means
the enabled accepted-step budget was exhausted while restoration was still
making progress. `linearization_infeasible` is retained when restoration is disabled.
QP errors/limits retain their existing `qp_failure`/`time_limit` semantics.
None of these statuses establishes nonlinear infeasibility.

`RunReport` adds `restoration_attempts`, `restoration_steps`,
`restoration_seconds` and `restoration_history`. Restoration time includes
callbacks and QP time and therefore overlaps those phase totals. Overall QP
counts include both phases. The optional `qp_trace` remains scoped to ordinary
SQP step models. With `record_history=True`, ordinary `IterationRecord` entries
stay in `history`; `RestorationRecord` entries describe accepted feasibility
steps in `restoration_history`. Their shared `iteration` numbers count all
preceding accepted steps, so ordinary history may contain gaps. Failed
restoration attempts are counted but have no accepted-step record.

A restoration record contains `iteration`, `objective_before/after`,
`primal_before/after`, `merit_before/after`, `step_length`, `trials` and
`qp_iterations`. No NLP residual is fabricated for these entries. With history
disabled both history vectors remain empty. A zero-Jacobian infeasible point
can remain stationary for this local restoration model; random perturbation,
second-order escape, scaling and global feasibility search are not implemented.


## Performance release 0.14.0

`SqpOptions.hessian_block_size` is 0 by default (full BFGS), or a contiguous
block width in 1..=256. The final block can be shorter. A width at least n is
identical to full mode. Variables stay in their existing callback order;
constraints, callbacks and tolerances do not change. Choosing a block width
approximates curvature and can change the outer trajectory. Choose it from
model structure and measured original-unit acceptance, not from a promised
universal speedup.

Each block uses damped BFGS and a fresh Cholesky check. On its first failed
update, if the existing Hessian-reset budget permits, the driver switches to
full BFGS, resets the Hessian and invalidates the QP template. The switch
consumes one normal Hessian reset and increments
`RunReport.block_hessian_fallbacks`. Limits are not increased. Full mode
continues for the rest of that solve. The block mode's initial orbit failures
and the validated hybrid results remain in the performance records.

Full BFGS does not factorize `B`: the QP adapter's exact gate, described
under the Clarabel adapter, is the only positive-definiteness test, once per
QP. (The delivered release maintained a rank-updated Cholesky factor above 16
variables. That factor tracked a nearby matrix, not the rounded `B` itself,
so an indefinite `B` reached the adapter and the solve failed with
`Unsupported`; see `crates/sqp/tests/hessian_gate.rs`.) The adapter also
checks accepted QP solutions in original coordinates.

`QpResidualWorkspace`, `QpModel.residuals_with_workspace`,
`QpModel.accepts_with_workspace`, and `kkt::residuals_with_workspace` reuse
owned buffers. They retain the allocating APIs' validation and acceptance
rules. Backend and driver have separate workspaces and independently repeat
the checks. Callback output buffers are refilled with NaN before each call,
including reused buffers; incomplete writes remain errors. Python callback
input lists retain their original fresh-list contract.

Conic conversion caches require exact CSR pattern and bound-side-class
compatibility. Every P/A/b value is refreshed, and q is read from the current
model. Only accepted optima retain conversion data. Clones begin with empty
caches. This numeric-data allocation cache also works when QP solver workspace
reuse is disabled; no primal/dual iterate warm start is introduced.
