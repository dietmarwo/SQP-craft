# Design notes

Scope: the SQP/NLP track of Planning2027 Idea 004 ("Open Rust solvers",
concept v3, 22 September 2026) and its parallelism companion (plan v2). This
repository covers only the nonlinear track; the LP/MIP interface is separate.
It defines contracts and test fixtures. It contains no optimizer, makes no
performance or parity claims, and does not depend on the competition workflow.

## Review of the 15 October interface draft

The draft (Rust core, PyO3 binding, typed Python facade, dummy backend, 6 Rust
and 21 Python tests) built and passed as delivered. Running it and comparing it
with the plan led to these changes:

| Draft | Change | Reason |
| --- | --- | --- |
| Four callbacks: `objective`, `constraints`, `gradient`, `jacobian_values`, each returning a new `Vec` | Two grouped callbacks, `functions(x) -> (f, c)` and `derivatives(x) -> (grad f, J values)`; Rust writes into solver-owned buffers | SQP always needs `f` with `c` (line search) and `grad f` with `J` (QP, BFGS). Trajectory models compute `f` and `c` in one propagation. Halves Python crossings; no per-call allocation; lengths fixed by construction |
| No relation between gradient and function calls | `derivatives` only at the point of the preceding `functions` call | Lets a model reuse its propagation without keyed caching |
| A start outside the variable bounds was evaluated | Rejected before any model call; solvers never evaluate outside the box | Trajectory models are often undefined there (negative mass or time of flight). Rejecting keeps the draft's "no silent repair" rule |
| `ProblemSpec` with public fields, re-validated on every solve; Python spec a mutable dataclass of six positional lists | Validated constructor, immutable afterwards; Python `ProblemSpec` is a keyword-only native class, validated at construction, comparable and picklable | Errors surface where the spec is built; one spec serves many solves (multistart) and can be sent to worker processes |
| Errors without context ("invalid bounds", "CSR endpoints"); `Display` printed `Debug` | Messages name the field and entry, e.g. `variable bounds: entry 1 has lower=5, upper=4` | Debuggability on large transcriptions |
| No multiplier convention; QP sign conventions deferred | One convention, `grad f = Jᵀλ + z` (HiGHS's), for NLP results and the QP seam; `Multipliers` in `SolveResult`; independent `kkt::residuals` | Results, warm starts and residuals depend on it; fixing it now costs nothing and the tests pin it with flipped-sign cases. The residual checker is the "independent evaluator" the plan asks for (milestone 2) |
| QP seam: no multipliers, every outcome other than success an error, primal-only warm start | `QpSolution` with multipliers and working set; `QpOutcome::{Optimal, Infeasible, Unbounded, LimitReached, Cancelled}`; `QpWarmStart` with step, multipliers and/or `ActiveSet`; `QpModel::residuals` | An SQP driver needs QP multipliers, must react to an infeasible linearization (elastic mode) rather than abort, and active-set warm starts are the comparison planned for milestone 3 |
| Python observer had to return a `bool`; `True` continued. `observer=print` raised `TypeError` after the evaluation | `True` stops; `None`/`False` continue; anything else is a `TypeError` | Print-only observers work; matches SciPy's convention that `True` means stop |
| No way to say "model undefined here" | `Error::UndefinedPoint` / `UndefinedPointError` | Lets a future line search backtrack, as SNOPT and Ipopt do, instead of aborting |
| Wrong shapes and non-finite outputs raised plain `RuntimeError` | `EvaluationError`, a `RuntimeError` subclass | Catchable specifically; compatible with the old behavior |
| An output entry the model forgot to write went unnoticed | Buffers are pre-filled with NaN | Unwritten entries are reported as non-finite |
| `x0=` or `warm_start=WarmStart(primal=...)`, exactly one | `x0` only | Two spellings of the same thing; a warm-start type returns when it carries more than a primal point |
| `include!("common.rs")` copy in the binding, archive checksums, `open-sqp` names, PyO3 0.27.2, edition 2021 | Proper modules, repository names (`sqp-craft`, `sqp_craft`), MIT per the repository LICENSE, PyO3 0.30.1, edition 2024, CI | Repository rather than archive |
| `Status`, `Error` exhaustive | `#[non_exhaustive]` | Real backends will add statuses without a breaking change |

Kept unchanged in substance: the dummy's evaluation-only behavior and its
refusal to claim convergence (`stub` even at the minimizer, `None` for absent
residuals); cancellation checkpoints and the "candidate only after a complete
evaluation" rule; `threads = 1` only and no silently ignored settings; a core
crate without dependencies; serial Python callbacks with the original
exceptions re-raised; the binding tested only through the compiled extension;
the central-difference check through the native boundary.

## Deliberately deferred

No setting is exposed before a backend honours it. Deferred until a real
backend or the real application needs them:

- exact Lagrangian Hessian callbacks, limited-memory or block quasi-Newton
  options, scaling and the corresponding multiplier transforms;
- iteration, evaluation and time budgets; a run record with phase timings
  (including GIL wait) and observed thread counts;
- QP backend capability negotiation (convexity, warm-start kinds, certificates,
  thread limits) and backend-specific warm-start state;
- batched evaluation, NumPy buffer exchange, and an evaluator flag declaring
  interpreter-bound evaluation (plan v2, section 6);
- Python bindings for the KKT residuals and the QP seam;
- an execution policy for independent parallel solves; free-threaded Python.

## Numerical boundary

All checks are floating-point checks, not exact certificates. Infinite values
are allowed only as `-inf` lower or `+inf` upper bounds; points, callback values
and matrix entries must be finite. Feasibility and residuals use absolute
tolerances in original units. That is a draft convention: mixed-unit trajectory
constraints may need per-row tolerances defined in the target card. A feasible
start is not convergence, an infeasible candidate does not prove infeasibility,
and a zero gradient alone does not certify a minimum. KKT residuals alone do
not exclude saddle points.

## Next steps

1. Fill `target-card.md` from the Python low-thrust application and freeze its
   reference data before any tuning.
2. Write the application's model against the two-callback protocol, unchanged
   otherwise, and run it through open nonlinear baselines (Uno all-open preset,
   Ipopt, SLSQP). Check every baseline's output with `kkt::nlp_residuals` after
   converting its multiplier signs.
3. Measure callback, copy and GIL costs on that model; then decide on NumPy
   buffers, batched evaluation or process-per-worker solves.
4. Native pilot (concept v3, milestone 3): damped BFGS with a convex QP backend
   behind `QpBackend` (HiGHS active-set, Clarabel as the interior-point
   comparison), recording QP sequences and checking each solution with
   `QpModel::residuals`. Use `UndefinedPoint` for line-search backtracking.
5. Robustness before speed (milestone 4): elastic mode or restoration driven by
   `QpOutcome::Infeasible`, scaling, globalization.
