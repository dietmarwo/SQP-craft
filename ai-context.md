# AI context for solving smooth constrained problems with SQP-craft

Last reviewed 2026-10-19 against `main` at `2578c11` (workspace 0.16.0,
including the shared agent job API 0.1, #12). In another checkout, check
`Cargo.toml`, `python/sqp_craft/__init__.py`, the worker's discovery output
and the linked guides before using a command or claiming a capability.

This file is operational context for an AI that must solve a user's smooth
nonlinear program with this repository, or decide that another tool fits
better. It is not a performance claim. A `converged` result is a
first-order (KKT) point under declared tolerances: it is neither a global
optimum nor a second-order certificate, and a failure is not a proof of
infeasibility. Report results that way.

## What the repository is

SQP-craft is a sequential quadratic programming solver in Rust with Python
bindings, aimed at smooth constrained problems such as trajectory
transcriptions. It is a research pilot for at most 256 variables: dense damped
BFGS, step QPs solved by Clarabel (interior point, the default) or by a dense
Goldfarb–Idnani active-set method, an ℓ1 merit line search, bounded
feasibility restoration and independent first-order KKT checks. No external
optimizer, BLAS or CMake is needed; Clarabel, nalgebra and Rayon are pinned
Rust dependencies.

| Package (crate or distribution) | Path | Use as |
|---|---|---|
| `sqp-craft` | `crates/core` | `sqp_craft`: problem spec, `NlpProblem` callbacks, solver trait, results, KKT residuals, QP seam, derivative checker, dummy method; no dependencies |
| `sqp-craft-solver` | `crates/sqp` | `sqp_craft_solver`: `NativeSqp<B>`, `SqpOptions`, `batch::solve_many` |
| `sqp-craft-clarabel` | `crates/clarabel` | `ClarabelQp`, the default QP backend |
| `sqp-craft-activeset` | `crates/activeset` | `ActiveSetQp`, the dense active-set backend |
| `sqp-craft-parallel` | `crates/parallel` | Explicit bounded execution resources (Rayon pools) for kernels and native models |
| `sqp-craft-python` / `sqp-craft` | `crates/python`, `python/sqp_craft` | `import sqp_craft`: PyO3 package |
| `optimization-job-api`, `optimization-runner` | `crates/` | Shared job protocol and the `optimization-run` runner |
| `sqp-job-worker` | `crates/sqp-job-worker` | Job worker for profile `sqp.solve/1` |

Nothing is published. The workspace sets `publish = false`, and neither
crates.io nor PyPI has an `sqp-craft` package at this review. Use a checkout
of https://github.com/dietmarwo/SQP-craft. The workspace version is 0.16.0;
the job crates are 0.1.0. `sqp.solve/1` is a protocol identifier, independent
of both. The "Version 0.4.0" line at the top of `docs/api.md` is stale; go by
`Cargo.toml`.

Toolchains:

- The solver crates need Rust 1.85 or newer; CI checks them with 1.85. The job
  crates need Rust 1.88, and CI checks them with 1.88.
- The Python package needs CPython 3.10 or newer and a C linker, and builds
  from source with Rust: `python -m pip install ".[test]"`. It is an abi3
  wheel; `--no-default-features` builds one without the native solver. CI tests
  3.10 and 3.14 on Linux, macOS and Windows.
- The job layer is Linux only.

The [application target card](docs/target-card.md) is intentionally
incomplete: it awaits the Python low-thrust trajectory application. The orbit
transfer fixture and generated problems are development tests. Do not claim
application-level performance.

## Required workflow for the AI

Before modelling, write a problem card with these facts. Inspect the user's
code and data where possible. Ask only for facts that cannot be inferred and
would change the solution.

| Question | Why it matters |
|---|---|
| Are the objective and constraints twice continuously differentiable where the solver may evaluate them? | SQP needs smoothness; kinks, branches and noise break BFGS and the line search. |
| How many variables? | The native method accepts at most 256 (`dense_size_limit`). |
| Which bounds and constraint rows apply, and which rows are equalities? | Bounds hold at every iterate; rows may be violated until convergence. |
| Are analytic or automatic derivatives available, and what is the Jacobian sparsity? | The model must supply the gradient and the Jacobian values for a fixed CSR pattern. |
| Which rows are linear? | A start that violates declared linear rows is projected onto them; later iterates stay on them, and they leave the merit function. |
| Where is the model undefined (failed propagation, negative mass)? | The model must signal it; the solver then shortens the trial step. |
| What start is available, and is it inside the bounds? | Starts are never projected; a start outside the box is rejected. |
| What are the units and scales of variables, objective and rows? | Tolerances are absolute and unscaled; very large gradients break the first step. |
| Which feasibility and stationarity tolerances does the application accept? | Defaults are 1e-7 (primal) and 1e-6 (stationarity, complementarity). |
| Is one local solution enough, or are several starts needed? | SQP is local; multistart is the way to explore. |
| What is the evaluation, time and core budget? Is a callback expensive? | Callback counts dominate cost; limits set the stopping rule. |
| Is there a reference solver or known solution? | Validate against it with an independent KKT check. |

For every completed solution, state the problem card, the backend and every
non-default option, the start (or start set), budgets, the termination status,
the independently recomputed objective, primal violation and KKT residuals,
and how the result was validated.

## First decide whether SQP-craft is appropriate

Use it for smooth constrained problems with at most 256 variables and reliable
derivatives, when a dependency-light Rust solver with independent KKT checks
is valuable, and when a local first-order solution is the goal.

Prefer another tool when:

- the problem is linear or mixed-integer linear: use
  [LP-IP-craft](https://github.com/dietmarwo/LP-IP-craft);
- the objective is a black box, nonsmooth, noisy or strongly multimodal, or
  gradients are unavailable: use
  [fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust), possibly as an
  outer global layer around local SQP refinement;
- there are more than 256 variables, or exact sparse second derivatives are
  available and matter: a sparse interior-point or exact-Hessian SQP code
  (the IPOPT, WORHP, KNITRO family) fits better; SQP-craft has no
  exact-Hessian option;
- the application needs a global optimum, a second-order certificate or a
  certificate of infeasibility: none is provided;
- the variables are integer: there is no integer support.

## Choose the integration surface

| Need | Entry point | Scope |
|---|---|---|
| Rust model | `NativeSqp<B>` with `ClarabelQp` or `ActiveSetQp` | Full option set, observer, cancellation |
| Python model | `sqp_craft.solve(..., method="sqp")` | Same solver; callbacks run on the calling thread |
| Many starts, Rust model | `sqp_craft_solver::batch::solve_many` / `try_solve_many` | Ordered results, worker-local models and backends |
| Many starts, Python model | Separate processes (the spec is picklable) | Python callbacks cannot run on solver threads |
| Durable, CPU-budgeted agent experiments | `optimization-run` + `sqp-job-worker` | Registered native models only; see below |
| Check derivatives or a KKT point | `check_derivatives`, `kkt_residuals` | Independent of any solver |

Python's `solve` defaults to `method="dummy"`, which only evaluates the start
and returns `stub`. Always pass `method="sqp"` to optimize.

## Formulate the problem

```text
minimize    f(x)
subject to  constraint_lower <= c(x) <= constraint_upper     (m rows, m >= 0)
            lower            <=  x   <= upper                (n variables, 1 <= n <= 256)
```

There is no hidden sign change: maximize by minimizing `-f`. Equal finite
bounds express equalities; infinite sides express one-sided or free entries.
All numbers are unscaled `f64`; no automatic scaling is applied unless
requested.

`ProblemSpec` holds the bounds and the fixed Jacobian sparsity pattern in
zero-based CSR (`indptr` with `m + 1` entries, strictly increasing column
indices per row). The pattern must contain every entry that can be nonzero
anywhere in the domain; it defines the order of the Jacobian values the model
returns. A `scipy.sparse` CSR matrix's `indptr` and `indices` can be passed
after `sort_indices()`. The spec is validated once, is immutable and, in
Python, picklable.

The model provides two callbacks:

| Rust (`NlpProblem`) | Python method | Computes |
|---|---|---|
| `functions(x, constraints_out) -> f` | `functions(x) -> (f, c)` | objective and constraint values |
| `derivatives(x, gradient_out, jacobian_out)` | `derivatives(x) -> (g, jac_values)` | dense gradient and Jacobian values in pattern order |

The solver guarantees that `x` is finite and inside the variable bounds, and
that `derivatives` is called only at the `x` of the immediately preceding
`functions` call, so a trajectory model can reuse one propagation. Signal an
undefined point with `Error::UndefinedPoint` (Python:
`raise sqp_craft.UndefinedPointError`): at a trial point the step is halved;
at the start the solve aborts. A model instance is mutable and local to one
solve.

Python:

```python
from sqp_craft import ProblemSpec, SqpOptions, check_derivatives, solve

class Toy:
    """minimize (x0-1)^2 + (x1-2)^2  subject to  x0 + x1 = 3,  0 <= x <= 4"""
    def functions(self, x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, [x[0] + x[1]]
    def derivatives(self, x):
        return [2 * (x[0] - 1), 2 * (x[1] - 2)], [1.0, 1.0]

spec = ProblemSpec(lower=[0, 0], upper=[4, 4],
                   constraint_lower=[3], constraint_upper=[3],
                   jacobian_indptr=[0, 2], jacobian_indices=[0, 1])
assert check_derivatives(spec, Toy(), [0.5, 0.5]).passed
result = solve(spec, Toy(), [0.5, 0.5], method="sqp",
               method_options=SqpOptions(qp_solver="active-set", elastic_mode=True))
assert result.status == "converged"
print(result.candidate, result.multipliers, result.stationarity)
```

Rust (adapted from `crates/sqp/examples/solve.rs`):

```rust,ignore
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result,
    SqpSolver, Status,
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

struct Toy;
impl NlpProblem for Toy {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x[0] + x[1];
        Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
        jac.fill(1.0);
        Ok(())
    }
}

let spec = ProblemSpec::new(
    Bounds::new(vec![0.0, 0.0], vec![4.0, 4.0]),
    Bounds::new(vec![f64::NEG_INFINITY], vec![2.0]),
    CsrPattern::dense(1, 2),
)?;
let mut solver = NativeSqp {
    qp: ActiveSetQp::default(),
    settings: SqpOptions { elastic_mode: true, ..SqpOptions::default() },
};
let r = solver.solve(&spec, &mut Toy, &[0.2, 0.4], &Options::default(),
                     &CancellationToken::default(), None)?;
assert_eq!(r.status, Status::Converged);
```

`NativeSqp::<ClarabelQp>::default()` selects the default backend. A sparse
pattern is `CsrPattern { rows, cols, indptr, indices }`.
`ProblemSpec::with_linear_rows(&rows)` (Python:
`ProblemSpec(..., linear_rows=[...])`) declares rows with a constant Jacobian.

Modelling rules:

- Run `check_derivatives` at representative points before solving. It
  compares every gradient and Jacobian entry, including entries missing from
  the pattern, with finite differences inside the bounds. `passed` needs no
  skipped coordinates.
- Scale variables, objective and rows to order one where possible. With
  ‖∇f(x₀)‖∞ of 10⁴ or more, the first line search often fails (the BFGS
  start is the identity); rescale, or try `initial_hessian_scale` near that
  gradient norm.
- Keep the box tight and meaningful: the model is never evaluated outside it.
- Declare linear rows only when they are exactly linear: a declared row whose
  Jacobian changes ends the solve with `InvalidInput`, and a wrong
  declaration can make `linear_infeasible` wrong.
- Use several starts for nonconvex problems and report the spread.

## Options

`Options` (Rust and Python) holds `feasibility_tolerance` (absolute, default
1e-7) and `threads` (maximum native workers, 1–8, default 1). Values above 8
are `Unsupported` (`NotImplementedError` in Python).

`SqpOptions` fields are all honoured; invalid values raise `InvalidInput`
(`ValueError`), and unknown names raise `TypeError` in Python. The ones that
matter most:

| Field | Default | Note |
|---|---:|---|
| `max_iterations` | 100 | Accepted steps |
| `max_callback_calls` | 2000 | `functions` plus `derivatives`, including rejected and undefined trials |
| `time_limit_seconds` | `None` | Cooperative whole-solve budget |
| `stationarity_tolerance`, `complementarity_tolerance` | 1e-6 | Absolute, original units |
| `dual_sign_tolerance` | 1e-8 | Multipliers whose sign selects an infinite bound |
| `qp_solver` | `"clarabel"` | Python only; `"active-set"` for the dense active-set backend (Rust: the type `B`) |
| `elastic_mode` | false | ℓ1-elastic step QPs instead of restoration; needs `"active-set"` |
| `max_restoration_steps` | 10 | Feasibility restoration budget; 0 disables restoration |
| `initial_hessian_scale` | 1 | `B = scale·I` at the start and after resets |
| `second_order_correction`, `merit="augmented-lagrangian"`, `hessian_update="augmented"`, `scaling`, `violation_limit`, `hessian_block_size` | off | Opt-in mechanisms; each helps on its own mechanism and costs elsewhere |
| `qp_reuse_workspace` | false | Reuses Clarabel allocations; no effect on the active-set backend |
| `record_history`, `record_qp_limit` | off | Iteration and QP traces in `report` |

Backend choice: the CUTEst benchmark of 1,036 problems
([docs/benchmark-results.md](docs/benchmark-results.md)) solved 754 with the
default Clarabel backend, 799 with `qp_solver="active-set"` and 813 with the
active-set backend plus `elastic_mode=True`, with no false convergence claim
in any configuration. It was measured with 0.14; the 0.16 review reran it
(Clarabel and elastic mode unchanged, active-set 801). The results recommend
active-set with elastic mode, but the defaults are unchanged until that is
decided. For a new problem, try both backends and report which one ran. On
the benchmark fixtures, the 164-variable orbit transfer among them, the
active-set backend takes the same steps as Clarabel and makes the whole solve
5.4–6.4× faster.

## Interpret results

| Status | Meaning |
|---|---|
| `converged` | All four first-order residuals within tolerance at the returned point |
| `iteration_limit`, `evaluation_limit`, `time_limit` | Budget exhausted; the last complete point is returned |
| `cancelled` | Token cancelled or the observer asked to stop; the last complete point is returned |
| `linearization_infeasible` | Inconsistent step QP with restoration disabled |
| `restoration_failed`, `restoration_limit` | Restoration reached a stationary point of the violation, or its step budget |
| `locally_infeasible` | Elastic mode: a stationary point of `f + γ·violation` with violated rows |
| `linear_infeasible` | The declared linear rows and the bounds have no common point (if the declaration is right) |
| `line_search_failure`, `no_descent`, `stalled` | No acceptable step, no merit descent even after a reset, or a step below the QP's accuracy |
| `qp_failure`, `numerical_failure` | QP limit or failure; BFGS failed after the reset budget |
| `stub` | Dummy method only |

None of the infeasibility statuses is a certificate. A `converged` status
says only that the first-order conditions hold at that point.

Results: `candidate` (or `None`), `evaluation` (`objective`, `constraints`,
`gradient`, `jacobian_values`, `primal_violation`, `feasible_candidate`),
`multipliers` (`constraints`, `variables`), the residuals `stationarity`,
`complementarity` and `dual_sign` (`None` when unavailable, never zero),
`counts` (`functions`, `derivatives`), `iterations`, `message` and `report`
(backend, QP and BFGS counters, restoration and elastic counters, timings,
optional history).

Multiplier signs follow the HiGHS convention: at a first-order point
`∇f(x) = J(x)ᵀλ + z`; a positive multiplier belongs to an active lower bound,
a negative one to an active upper bound, zero to an inactive one. After a
limit, the multipliers come from the previous iterate's QP; residuals
computed with any multipliers still bound the best first-order residual at
that point from above.

Errors: `InvalidInput` (`ValueError`) for malformed specs, starts or options;
`Evaluation` (`EvaluationError`) for non-finite or wrongly sized model output;
`UndefinedPoint` (`UndefinedPointError`) at the start; `Unsupported`
(`NotImplementedError`) for valid but unimplemented requests, such as
`n > 256`, `threads > 8` or elastic mode with Clarabel; `Backend`
(`RuntimeError`). Exceptions raised by Python callbacks are re-raised
unchanged.

### Independent checks

`kkt::nlp_residuals(&spec, x, &evaluation, &multipliers)` (Python:
`kkt_residuals(spec, x, evaluation, multipliers)`) recomputes the primal,
stationarity, complementarity and dual-sign residuals in unscaled units,
calling no model code. For an independent check, evaluate the returned
candidate with a fresh model instance, build the `Evaluation` from those
values, and pass the solver's multipliers; this is what the job worker does.

## Parallelism, cancellation and observers

- `Options.threads` is a maximum. Active-set kernels and dense outputs split
  work only when it is large (from 512 variables or 32,768 bounded-row
  entries); full BFGS stays scalar, and Clarabel's engine is serial. For
  typical NLPs of at most 256 variables, `threads` above 1 changes little.
- Multistart is the useful parallelism. In Rust, `batch::solve_many::<B, M, _>(
  spec, starts, settings, options, workers, factory, cancel)` constructs each
  model on its worker, reuses a backend per worker and returns ordered
  per-start results; `workers * options.threads <= 8`. Use eight outer
  workers with `threads = 1` for small NLPs. Cancellation keeps completed
  results; per-solve time limits are not a batch deadline.
- In Python, run independent solves in separate processes with their own
  model instances. The binding releases the interpreter for native work but
  calls every callback on the calling thread.
- Native models may implement `NlpProblem::set_threads` to split their own
  propagation with `sqp_craft_parallel::Execution` within the declared budget.
- `CancellationToken` / `Cancellation` is checked before every model call and
  inside the QP backend; it cannot interrupt a running callback. Ctrl-C ends a
  Python solve with `KeyboardInterrupt`. An observer receives `Progress`
  (iteration, objective, primal violation) and can stop the solve, which ends
  `cancelled`.

## Shared agent job API

Start with [the shared guide](docs/job-api.md) and
[the SQP tutorial](tutorials/job-api/README.md). The job layer gives an agent
durable, validated, CPU-admitted multistart solves with journals,
cancellation and result manifests. Its profile is deliberately smaller than
the in-process API.

```sh
cargo build --release --locked -p optimization-runner -p sqp-job-worker
python3 tutorials/job-api/run.py --root /tmp/sqp-job-tutorial
./target/release/optimization-run --root /tmp/sqp-job-tutorial \
  problems describe sqp
```

The tutorial creates an eight-slot node, registers the worker as `sqp`, and
runs the simplex-constrained quadratic (objective 0.25 at `x = 0.25`) and
Rosenbrock with the active-set backend and a shifted quadratic with Clarabel,
eight starts each. Commands
for a single job are in the guide.

### Registered models only

Jobs cannot carry Python or other callbacks. `sqp.solve/1` accepts only the
models compiled into the worker, and `inputs` must be empty:

| Problem ID | Model | Bounds and dimensions |
|---|---|---|
| `benchmark.quadratic` | Sum of `(x[i]-(i+1)/n)^2` | `[-5,5]`, 1–256 variables |
| `benchmark.simplex` | Sum of `x[i]^2`, subject to `sum(x) = 1` | `[0,1]`, 1–256 variables |
| `benchmark.rosenbrock` | Adjacent-pair Rosenbrock | `[-5,5]`, 2–256 variables |

A user's model therefore needs either the in-process API or a new worker
model: extend `Model` and `registered` in `crates/sqp-job-worker/src/main.rs`
(or write a separate worker with the shared SDK), with a validated
`ProblemSpec`, analytic derivatives and a finite-difference test like
`registered_derivatives_match_finite_differences`. Data files need hashed
inputs and a new profile version, because `sqp.solve/1` rejects inputs. Never
deserialize executable callbacks. The worker reports the engine version as the
literal `"0.16.0"`; update it with the workspace version.

### Job specification

`payload_schema` is `sqp.solve/1`. All payload fields are required, and
unknown fields are rejected:

| Payload field | Meaning |
|---|---|
| `problem`, `dimension` | Registered problem ID and its dimension |
| `starts` | One complete finite start per task, inside the model box (never projected) |
| `backend` | `active_set` or `clarabel` (with an underscore, unlike Python's `"active-set"`) |
| `max_iterations` | Accepted steps per task; 0 permits only the initial evaluation |
| `max_callback_calls` | Positive per-task function plus derivative calls |
| `feasibility_tolerance`, `stationarity_tolerance` | Positive finite values |

All other `SqpOptions` keep their defaults, so elastic mode is off even with
`active_set`. The remaining wall time is each task's cooperative time limit,
and `inner_threads` (1–8) maps to `Options::threads`. For the registered
models `inner_threads` above 1 changes nothing; use `8 / 8 / 1`
(`cpu_slots / outer_workers / inner_threads`) for multistart throughput.
The runner requires `outer_workers * inner_threads <= cpu_slots <= 1024`.
Validation echoes the specification and its output is capped at 4 MiB, so
about 600,000 one-dimensional starts fit, not the 1,000,000 tasks the
envelope allows.

### Job results

Inspect the job `state` first: `completed` only means every task returned.
Then read each outcome with `optimization-run result JOB_DIRECTORY`:

| Outcome field | SQP meaning |
|---|---|
| `termination` | The native status |
| `claim` | `first_order` (native `converged` plus the worker's own KKT recheck), `feasible` (primal feasible, no first-order claim), or `none` |
| `feasible`, `objective`, `x` | From a fresh model evaluation at the candidate |
| `details` | `kkt` residuals, `qp_backend`, `qp_status`, `message`, `global_optimum_certified: false`, and the `acceptance` thresholds |

The recheck uses `feasibility_tolerance`, `stationarity_tolerance`,
complementarity ≤ 1e-6 and dual sign ≤ 1e-8. A native `converged` that fails
it fails the job instead of being reported. Work counters are `functions`,
`derivatives`, `iterations`, `qp_iterations` and `validation_calls`; never add
them to another profile's counters. Graceful cancellation sets the native
token. Killed workers leave `work` null and only committed tasks count; there
is no BFGS or solver checkpoint resume.

### Shared copies

The shared crates and `docs/job-api.md` were delivered as identical copies in
fcmaes-rust, LP-IP-craft and SQP-craft. They now differ, and this repository
has the most runner fixes. Its review ([docs/job-api/REVIEW.md](docs/job-api/REVIEW.md))
changed nine shared files: an unreadable journal seals the job `failed` with
`runner.journal_error` instead of breaking `jobs list`; a job whose launcher
died while queued can be repaired; finished jobs are no longer flagged after
a reboot; staging uses a `.staging-*` directory under the node root, so
`TMPDIR` may be `noexec` but the root must allow execution; a signal during
validation stops the validator; an append after a torn journal tail no longer
destroys earlier records; `./model.json` is a valid input path; and detached
units are named `optimization-job-ID.service`. Of these, fcmaes-rust
(`bf68a16`) fixed only the queued-launcher repair, in a different way;
LP-IP-craft has the delivered code.

The worker protocol is unchanged across the copies. This runner drove the
LP-IP-craft and fcmaes-rust tutorial jobs, and the SQP worker ran under the
fcmaes-rust runner, each on one node with all three workers (checked
2026-10-19). Run a node root through one runner build; mixing builds
concurrently is untested. `sha256sum -c docs/job-api/SHARED-SOURCES.sha256`
checks this repository's copies, not the siblings'. The raw validation logs of
the delivery are under `validation/job-api/`.

## Validation and reporting checklist

1. Check derivatives with `check_derivatives` before trusting any solve.
2. Recompute the objective, constraint values and primal violation at the
   candidate with a fresh model, and the KKT residuals with `kkt_residuals`,
   in original units.
3. Compare with a reference where one exists: SciPy SLSQP under the same
   check (`examples/compare_baseline.py`), the CUTEst harness under `bench/`,
   or the user's own solver. A comparator's own success flag is not evidence:
   154 of SLSQP's CUTEst successes failed this check.
4. Report the status, residuals, objective, counts, iterations, backend, all
   non-default options, tolerances, starts and wall time.
5. Say "first-order point under tolerance", not "optimal". For
   `restoration_failed`, `locally_infeasible` and the other failures, say the
   solver did not find a feasible point from that start, not that none exists.
6. For tuning, freeze tolerances and held-out problems first, change one
   option at a time, and keep regressions in the report.

## Failure diagnosis

| Symptom | Likely cause and action |
|---|---|
| `line_search_failure` at iteration 0 | Huge initial gradient or bad scaling; rescale, or raise `initial_hessian_scale` |
| `qp_failure` | Degenerate or inconsistent QPs; try the other backend, elastic mode with the active-set backend, or rescaling |
| `restoration_failed` / `restoration_limit` / `locally_infeasible` | Probably an infeasible start region or an infeasible model; try other starts, check the constraint formulation |
| `iteration_limit` | Slow BFGS convergence; raise the budget, improve scaling or the start |
| `stalled` / `no_descent` | Noise-level steps or inaccurate derivatives; rerun `check_derivatives`, tighten `qp_tolerance` only with evidence |
| `numerical_failure` | BFGS reset budget spent; check derivatives and scaling |
| `EvaluationError` | Non-finite or wrongly sized model output; fix the model or signal undefined points |
| `NotImplementedError` | `n > 256`, `threads > 8`, elastic mode with Clarabel, or a solver not compiled in |
| A converged run is far from the expected solution | A different local solution; use more starts |

## Test commands

The CI set, from the repository root (`scripts/check.sh` runs the main Rust
and Python checks together):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo run --locked -p sqp-craft --example toy
cargo run --locked -p sqp-craft-solver --example solve
cargo test --release --locked -p optimization-job-api -p optimization-runner -p sqp-job-worker
cargo build --release --locked -p optimization-runner -p sqp-job-worker
python3 scripts/job-api/test_integration.py
python3 scripts/job-api/test_review.py
python3 tutorials/job-api/run.py --root /tmp/job-api-tutorial
python -m pip install ".[test]" && python -m pytest -q
python examples/solve_sqp.py
```

CI also runs the SLSQP comparison and the random, complementarity and SNOPT
stress families, which fail on any false convergence claim, a 20-problem
CUTEst subset, and a build without the native solver. Pass `--fcmaes` and
`--lp-ip` executables to `test_integration.py` to test a mixed node.

## Repository references

- [README.md](README.md): release notes, Python usage, build and tests
- [docs/api.md](docs/api.md): the contract: problem, callbacks, options, statuses, errors, KKT, QP seam, batches
- [docs/benchmark-results.md](docs/benchmark-results.md): CUTEst results and failure classes
- [docs/gaps.md](docs/gaps.md): weaknesses compared with mature SQP codes
- [docs/active-set-qp.md](docs/active-set-qp.md), [docs/elastic-mode.md](docs/elastic-mode.md),
  [docs/second-order-correction.md](docs/second-order-correction.md),
  [docs/snopt-lessons.md](docs/snopt-lessons.md): the opt-in mechanisms
- [docs/performance-60/REPRODUCE.md](docs/performance-60/REPRODUCE.md): thread budgets and batches
- [docs/job-api.md](docs/job-api.md), [tutorials/job-api](tutorials/job-api/README.md),
  [docs/job-api/REVIEW.md](docs/job-api/REVIEW.md): shared job API and this worker
- [docs/target-card.md](docs/target-card.md): the open application contract
- `crates/core/src/kkt.rs`, `crates/core/src/derivative_check.rs`: independent checks
- `crates/sqp-job-worker/src/main.rs`: the `sqp.solve/1` adapter and registered models
- `python/sqp_craft/__init__.py`: Python API and defaults
