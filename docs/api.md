# Contract: problem, callbacks, solver, results

Version 0.2.0. Interfaces are pre-release; `publish = false` prevents
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
  A trial point that roundoff would push outside is shortened, never clipped.
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
implemented by `DummySolver` and `NativeSqp<ClarabelQp>` (crates `sqp-craft`,
`sqp-craft-solver`, `sqp-craft-clarabel`).

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
| `qp_tolerance` | `1e-10` | Clarabel's native tolerances |
| `qp_max_iterations` | `200` | Per QP |
| `record_qp_limit` | `0` | Step QPs kept in `report.qp_trace`, at most 16 |
| `record_history` | `false` | One `IterationRecord` per iteration in `report.history` |

## The native SQP method

A dense-BFGS line-search SQP for at most 256 variables. Each iteration at `x_k`:

1. **Step QP.** `min ½pᵀBp + gᵀp` subject to
   `lower_c − c(x_k) <= J p <= upper_c − c(x_k)` and
   `lower − x_k <= p <= upper − x_k` (bounds shifted to step coordinates),
   solved by the QP backend and accepted only through `QpModel::accepts`
   (below). A step component of a fixed variable within the feasibility
   tolerance of zero is set to exactly zero, counted in
   `fixed_step_corrections`, and the QP is rechecked.
2. **Convergence test** at `x_k` with the QP's fresh multipliers: `converged`
   only if primal violation, stationarity, complementarity and dual-sign
   residuals are all within their tolerances. A small step, a small objective
   change or QP success alone never ends the solve.
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
   ends; a symmetric positive-definite check (Cholesky) guards every update,
   and a failed update resets `B` to `scale·I`.

If the predicted decrease is not positive, `B` is reset once and the QP
retried. If it is still not positive and its magnitude is within the QP's own
accuracy (interior-point solutions stop slightly short of active bounds, which
costs up to the QP complementarity residual per entry), the step is QP noise and
the solve ends `stalled`; otherwise it ends `no_descent`.

There is no restoration or elastic mode, no second-order correction (so the
Maratos effect can cost backtracking steps), no automatic scaling, and no
exact-Hessian option. First-order convergence is not a global or second-order
optimality certificate.

### Statuses

| Status | Meaning |
| --- | --- |
| `converged` | All four first-order residuals within tolerance at the returned point |
| `iteration_limit`, `evaluation_limit`, `time_limit` | Budget exhausted; the last complete point is returned |
| `cancelled` | Token cancelled or observer stop; the last complete point is returned |
| `linearization_infeasible` | The step QP's linearized rows are inconsistent; the nonlinear problem may still be feasible |
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
iterations, rejected and undefined trials, Hessian resets, fixed-step
corrections, the last accepted step norm, inclusive phase times (total,
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
acceptance tolerance, iteration cap, remaining time) and returns a `QpOutcome`:
`Optimal(solution)`, `Infeasible`, `Unbounded`, `LimitReached(last iterate)` or
`Cancelled`; `report()` gives the backend name, raw status, limit kind and
iterations. Infeasibility and unboundedness are outcomes, not errors. A
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

`DummyQp` validates its input and returns `Unsupported`; it never fabricates a
step.

### Clarabel adapter (`sqp-craft-clarabel`)

Clarabel 0.11.1, an interior-point conic solver in Rust. Cold starts only:
supplied warm-start data is validated, then rejected as `Unsupported` rather
than ignored. One thread, positive-definite Hessians (checked by Cholesky), at
most 256 variables. Each finite side of a row or variable bound becomes one
nonnegative-cone row (`a·p ≤ u` as `(a, u)`, `a·p ≥ l` as `(−a, −l)`); equal
bounds become a zero-cone row; free sides produce no row. Clarabel's duals
`z ≥ 0` satisfy `Hp + g + Aᵀz = 0`, so `λ = z_lower − z_upper`, and `−z` for an
equality. A Clarabel iteration callback polls the cancellation token and the
deadline. A `Solved` result is kept even if the deadline passed during the
solve; reduced-accuracy (`AlmostSolved`) and limited iterates are never used as
steps. HiGHS was probed first (see the results document); it lacks a user
interruption callback in its QP solver.

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
