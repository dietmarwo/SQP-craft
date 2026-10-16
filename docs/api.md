# Contract: problem, callbacks, results

Interface draft v0.1.0. Names and signatures are provisional until the first
real backend is integrated; `publish = false` prevents accidental releases.

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
Python: `solve(spec, model, x0, *, options=None, cancellation=None, observer=None)`.

The start point `x0` must be finite and inside the variable bounds; it is
rejected, never projected or repaired. Input errors are reported before any
model call and take precedence over a pending cancellation.

`Options` exposes only settings a backend honours: `feasibility_tolerance`
(absolute, unscaled; default `1e-7`) and `threads` (only `1`; anything else is
`Unsupported`/`NotImplementedError`). Iteration limits, scaling, Hessian
choices and evaluation budgets will arrive with the backend that implements
them, so no setting is silently ignored.

## Results

| Field | Meaning |
| --- | --- |
| `status` | `stub` or `cancelled` (more with real backends; `Status` is `#[non_exhaustive]`) |
| `candidate` | the returned point, or `None` if no complete evaluation happened |
| `evaluation` | `f`, `c`, `grad f`, Jacobian values, `primal_violation` and `feasible_candidate` at the candidate |
| `multipliers` | estimates in the convention below, when the backend computes them |
| `counts` | `functions` and `derivatives` calls, including discarded ones |
| `iterations` | completed SQP iterations |
| `stationarity`, `complementarity` | residuals at the candidate, when multipliers exist |
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
(elastic mode or restoration; Hessian regularization). A `QpSolution` carries the
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
