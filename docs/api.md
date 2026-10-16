# Contract: problem, callbacks, results

Iteration 1 v0.2.0. Interfaces remain pre-release; `publish = false` prevents accidental releases.

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

**Guarantees from every solver in this crate**

- `x` has length `n`, is finite and lies within the variable bounds. The model
  is never evaluated outside the box, so it may rely on, say, a positive mass.
- `derivatives` is only called at the `x` of the immediately preceding
  `functions` call, so intermediate results may be reused.
- Rust output buffers have exactly the required length; they are pre-filled with
  NaN, so an entry the model forgets to write is reported as non-finite.
- Python callbacks receive a fresh `list[float]`; mutating it has no effect on
  the solver.

**Model obligations**

- Return finite values with the right lengths. Python callbacks may return
  lists, tuples or 1-D NumPy arrays; the two results must be returned as a
  `tuple`.
- Signal "not defined at this point" (a failed propagation, say) with
  `Error::UndefinedPoint` / `raise sqp_craft.UndefinedPointError`. Line searches
  will treat it as a request to shorten the step; at the start point it aborts
  the solve.
- Any other error aborts the solve.

The callback trait requires neither `Send` nor `Sync`: a model instance is
mutable and local to one solve. Parallel strategies run independent solves with
independent model instances (see design notes).

## Solving

Rust: `SqpSolver::solve(&mut self, spec, problem, x0, options, cancel, observer)`.
Python: `solve(spec, model, x0, *, options=None, cancellation=None, observer=None, method="dummy", method_options=None)`.

`method="sqp"` selects `NativeSqp<ClarabelQp>`. `method_options` is a
`SqpOptions` value and is rejected for the dummy.

The start point `x0` must be finite and inside the variable bounds; it is
rejected, never projected or repaired. Input errors are reported before any
model call and take precedence over a pending cancellation.

`Options` exposes only settings a backend honours: `feasibility_tolerance`
(absolute, unscaled; default `1e-7`) and `threads` (only `1`; anything else is
`Unsupported`/`NotImplementedError`). Native iteration, callback and time budgets live in `SqpOptions`, below.
There is no automatic model scaling, user Hessian or parallel callback mode.

## Results

| Field | Meaning |
| --- | --- |
| `status` | `stub`, `cancelled`, `converged`, limits and explicit pilot failure statuses (below) |
| `candidate` | the returned point, or `None` if no complete evaluation happened |
| `evaluation` | `f`, `c`, `grad f`, Jacobian values, `primal_violation` and `feasible_candidate` at the candidate |
| `multipliers` | estimates in the convention below, when the backend computes them |
| `counts` | `functions` and `derivatives` calls, including discarded ones |
| `iterations` | completed SQP iterations |
| `stationarity`, `complementarity`, `dual_sign` | residuals at the candidate, when multipliers exist |
| `message` | explanation of the status |

`primal_violation` is the largest bound violation over variables and
constraints, absolute and in original units. `feasible_candidate` compares it
with the tolerance. Neither is evidence of optimality, and an infeasible
candidate does not show that the problem is infeasible.

### The dummy backend, exactly

1. Validate options and `x0`.
2. Check cancellation; call `functions(x0)` once; check outputs.
3. Check cancellation; call `derivatives(x0)` once; check outputs.
4. Check cancellation; report iteration 0 to the observer; check cancellation.
5. Return `x0` unchanged, `status = stub`, `iterations = 0`, and
   `multipliers`, `stationarity`, `complementarity` all `None`.

`stub` is returned even at the exact minimizer: the dummy never claims
convergence. A cancellation before step 3 completes returns `cancelled` with no
candidate but with the call counts so far. An observer stop or a cancellation
after step 3 returns `cancelled` with the complete evaluation.

## Multiplier sign convention and KKT residuals

At a first-order point

```text
grad f(x) = J(x)^T λ + z
```

where `λ` (one per constraint row) and `z` (one per variable) are the
multipliers; the Lagrangian is `f - λᵀc(x) - zᵀx`. A positive multiplier belongs
to an active lower bound, a negative one to an active upper bound, zero to an
inactive one. Equalities accept either sign. This is the HiGHS convention, and
the QP seam uses the same one, so QP multipliers are directly the next NLP
estimates.

`kkt::residuals` / `kkt::nlp_residuals` recompute, independently of any solver
and in unscaled units:

| Residual | Definition |
| --- | --- |
| `primal` | largest bound violation of variables and rows |
| `stationarity` | `‖grad f − Jᵀλ − z‖∞` |
| `complementarity` | largest `|μ| · |distance to the bound selected by sign(μ)|` |
| `dual_sign` | largest `|μ|` whose sign selects an infinite bound |

The tests pin the convention on problems with known solutions, including the
flipped-sign cases.

## Errors

| Rust `Error` | Python exception | When |
| --- | --- | --- |
| `InvalidInput` | `ValueError` | malformed spec, start point or options |
| `Evaluation` | `sqp_craft.EvaluationError` (a `RuntimeError`) | non-finite or wrongly sized model output |
| `UndefinedPoint` | `sqp_craft.UndefinedPointError` (an `EvaluationError`) | model not defined at the point |
| `Unsupported` | `NotImplementedError` | valid but unimplemented request |
| `Backend` | `RuntimeError` | backend setup/execution failure outside numerical driver handling |
| — | `TypeError` | wrongly typed callback results (not a 2-tuple, text, `None`, ...) |

Exceptions raised by Python callbacks or the observer are re-raised unchanged,
with their original type and traceback. Errors never carry partial results.

## Cancellation and observers

`CancellationToken` (Rust) / `Cancellation` (Python) is a shared flag checked
before the first model call, between calls and after the observer. It cannot
interrupt a running callback. A callback may cancel its own token; another
Python thread can only cancel when it gets interpreter time. Use process
isolation where stronger interruption is needed.

The observer receives `Progress(iteration, objective, primal_violation)`;
iteration 0 is the start point, not a step. In Rust it returns
`Control::Continue` or `Control::Stop` (closures implement `Observer`). In
Python it returns `True` to stop, `None` or `False` to continue, so
`observer=print` works. A stop ends the solve with `cancelled` and keeps the
candidate.

## QP subproblem seam (`sqp_craft::qp`, Rust only, provisional)

`QpModel` is the convex QP in the step `p`:

```text
minimize    0.5 pᵀHp + gᵀp
subject to  row_lower <= A p <= row_upper,   variable_lower <= p <= variable_upper
```

All bounds apply to the **step**: at iterate `x_k` they are the NLP bounds minus
`c(x_k)` or `x_k`. `H` is stored as its lower triangle in CSR; an off-diagonal
entry stands for both symmetric positions and is not doubled. Validation checks
structure, not positive semidefiniteness: the SQP driver must convexify `H`
before calling a convex backend.

A `QpBackend` returns a `QpOutcome`: `Optimal(solution)`, `Infeasible`,
`Unbounded`, `LimitReached(last iterate)` or `Cancelled`. Infeasibility and
unboundedness are outcomes, not errors, because the SQP driver reacts to them
(iteration 1 stops explicitly; restoration is deferred). A `QpSolution` carries the
step, multipliers in the shared convention, an optional working set
(`ActiveSet` of `Activity` per variable and row) and the iteration count.
`QpWarmStart` may carry a step, multipliers and/or a working set; each backend
uses what its method can exploit. `QpModel::residuals` checks a solution from
the model data alone. `DummyQp` validates its input and returns `Unsupported`;
it never fabricates a step, and the dummy NLP backend does not call it.

## Threading and Python

The core crate has no dependencies. The binding runs Python callbacks serially
on the calling thread with the interpreter attached, and never hands a Python
object to another thread. Data crosses the boundary by copy; this is a
correctness-first binding, not a zero-copy array API. Its overhead should be
measured on the real model before deciding on NumPy buffers, batched
evaluation or process isolation.

## Native method controls and behavior (0.2.0)

| `SqpOptions` | Default | Meaning |
| --- | --- | --- |
| `max_iterations` | `100` | Accepted SQP step cap |
| `max_backtracks` | `25` | Halvings after the first trial |
| `max_callback_calls` | `2000` | Sum of grouped functions and derivatives calls, including failures/rejections |
| `time_limit_seconds` | `None` | Cooperative whole-call budget after input validation |
| `stationarity_tolerance`, `complementarity_tolerance` | `1e-6`, `1e-6` | Separate unscaled KKT thresholds |
| `dual_sign_tolerance` | `1e-8` | Sign/residual threshold |
| `dense_size_limit` | `256` | Configurable cap from 1 through 256 variables |
| `initial_hessian_scale` | `1` | Positive diagonal initialization/reset scale |
| `max_hessian_resets` | `3` | Total bounded curvature/descent resets |
| `qp_tolerance` | `1e-10` | Clarabel solve tolerance |
| `qp_max_iterations` | `200` | QP iteration cap per subproblem |
| `record_qp_limit` | `0` | Opt-in original step-QP trace, at most 16 models |

Tolerance/scale fields must be positive and finite; time is nonnegative and
finite. Zero outer iterations evaluates the start but returns `iteration_limit`;
zero callback budget performs no callback. The core `Options` feasibility
tolerance remains the primal KKT threshold.

The loop builds shifted step bounds, solves and checks the QP, and recomputes
current NLP KKT residuals with fresh multipliers. It accepts `converged` only
when primal, stationarity, complementarity and dual-sign tests all pass.
The L1 merit penalty is nondecreasing, at least `1.1*norm_inf(lambda)+1`;
Armijo constant is `1e-4` and trial steps halve. BFGS uses the same new signed
multipliers at both endpoints, Powell damping and an SPD check. A non-descent
prediction permits one Hessian reset/retry within the global reset budget.

Statuses: `iteration_limit`, `evaluation_limit`, `time_limit`, `qp_failure`,
`linearization_infeasible`, `line_search_failure`, `no_descent`, `stalled`,
`numerical_failure`. Observer stop remains `cancelled`. An infeasible step QP
does not prove NLP infeasibility. A small step alone never proves convergence.

Trials stay strictly inside variable bounds. Fixed-variable step roundoff may
be set to exact zero only within feasibility tolerance; this is counted and the
entire QP is checked again. Other out-of-box steps are shortened or fail.
An undefined trial consumes a call and backtracks, clearing any saved recoverable
Python exception. Fatal exceptions preserve object identity/type/traceback.
If a budget stops a partial trial, results keep the previous fully evaluated
accepted point. Callbacks themselves cannot be forcibly interrupted.

`RunReport` adds method/backend/raw QP status, QP solves/iterations,
rejected/undefined trials, Hessian resets, fixed-step corrections, last accepted
step norm, total/callback/QP/BFGS times and optional bounded QP trace. Phase
measurements do not cover every miscellaneous driver operation. The termination
`message` supplies the reason; unavailable residuals remain `None`.

Clarabel 0.11.1 accepts SPD Hessians up to the dense cap. Lower CSR Hessian
entries convert once to upper CSC without doubling. Lower/upper/equality cone
rows reverse-map to the public signed multipliers. Its iteration callback polls
cancellation and the remaining deadline. Reduced-accuracy and limited iterates
are not accepted as steps. Supplied primal/dual/active-set QP warm starts are
validated then rejected as unsupported; the driver intentionally starts cold.

The Python adapter owns callback references, detaches for native computation,
and reattaches on the same calling thread for each grouped callback/observer.
No Python-bound reference spans a detached computation, and no Send/Sync bound
is imposed on a user's Rust `NlpProblem`. Separate process workers are supported.

## Derivative and residual tools

`check_derivatives(spec, model, x, *, steps=(1e-4,1e-5,1e-6),
absolute_tolerance=1e-5, relative_tolerance=1e-4, max_directions=256)` checks
coordinate directions using central differences inside the box and one-sided
differences at a boundary. It compares the objective and every row, including
entries omitted from the declared Jacobian pattern. It retains the best
resolvable estimate across step sizes and reports component errors. Fixed,
unrepresentable and unvisited coordinates are skipped explicitly and make the
report inconclusive, not passed. Checker callback counts are separate from
solve counts. Domain errors during a derivative check propagate.

`kkt_residuals(spec, x, evaluation, multipliers)` exposes the independent
original-unit Rust checker. No callback is invoked by it. `Bounds.checked_violation`
and all KKT products/differences reject nonfinite intermediate arithmetic.
