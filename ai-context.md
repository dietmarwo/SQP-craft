# AI context for solving smooth constrained problems with SQP-craft

Last reviewed 2026-10-20 against `main` at `d916290` (workspace 0.16.0). That
state includes the result contract 0.2 (claims with evidence,
`sqp.evaluate/1`, error codes, JSON Schemas, #15); the shared job crates taken
from [optimization-jobs](https://github.com/dietmarwo/optimization-jobs) at
`eb1c9de` instead of in-repo copies (#16, #17); the MCP server
`optimization-mcp` (#18); and refutation of failed rechecks instead of a
failed job, plus the tutorial's `--bin-dir` (#19). #20 moved the pin to
`6a42472` and the release preparation to `423cd87`, which add worker-declared
artifacts and paged artifact reads (`get_artifact` with `offset` and `limit`
also pages through the `results` journal of a running job); the pin is now
`43ef3ae`, which adds the MCP tools `put_input` and `get_guide` (with the
corrected Pareto guidance) and a host report in `node doctor`. In another
checkout, check `Cargo.toml`, `python/sqp_craft/__init__.py`, the worker's
discovery output and the linked guides before using a command or claiming a
capability.

Since that review: application models through the worker's public registry
(`sqp_job_worker::Registry`, a standalone worker in
`crates/sqp-job-worker/examples/custom_worker.rs`), the optional payload
fields `problem_version`, `config` and `elastic_mode` with model data
`inputs`, and the discovery keys `payload_schemas` and `examples` in the
worker's `describe` output.

This file is operational context for an AI that must solve a user's smooth
nonlinear program with this repository, or decide that another tool fits
better. It is not a performance claim. A `converged` result is a
first-order (KKT) point under declared tolerances: it is neither a global
optimum nor a second-order certificate, and a failure is not a proof of
infeasibility. Report results that way.

If the solver is not chosen yet, start with the
[entry point](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)
in optimization-jobs. It routes between fast-cma-es, fcmaes-rust,
LP-IP-craft and SQP-craft and covers installation, the job API and the MCP
server.

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
| `optimization-job-api`, `optimization-runner` | git dependency on [optimization-jobs](https://github.com/dietmarwo/optimization-jobs) | Shared job protocol and the `optimization-run` runner |
| `sqp-job-worker` | `crates/sqp-job-worker` | Job worker library (`Registry`, `Adapter`) and the stock binary, profiles `sqp.solve/1` and `sqp.evaluate/1` |

Nothing is published yet: neither crates.io nor PyPI has an `sqp-craft`
package at this review. The five library crates and the Python package are
prepared for both registries ([RELEASING.md](RELEASING.md)); the worker and
the benchmarks stay source-only. Use a checkout
of https://github.com/dietmarwo/SQP-craft. The workspace version is 0.16.0;
the shared job crates are 0.2.0, and the worker crate shares the workspace
version, which it reports as its engine version. `sqp.solve/1` is a protocol
identifier, independent of both. The "Version 0.4.0" line at the top of `docs/api.md` is stale; go by
`Cargo.toml`.

Toolchains:

- The solver crates need Rust 1.85 or newer; CI checks them with 1.85. The job
  worker and the shared job crates need Rust 1.88; CI checks the worker with
  1.88.
- The Python package needs CPython 3.10 or newer and a C linker, and builds
  from source with Rust: `python -m pip install ".[test]"`. The wheel is abi3.
  `python -m maturin build --release --no-default-features` (or
  `MATURIN_PEP517_ARGS=--no-default-features` with pip) builds one without the
  native solver. CI tests 3.10 and 3.14 on Linux, macOS and Windows.
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
| What start is available, and is it inside the bounds? | Starts are never projected into the box; a start outside it is rejected. Only declared linear rows can move a start. |
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

- the problem is an LP or a mixed-integer linear program: use
  [LP-IP-craft](https://github.com/dietmarwo/LP-IP-craft) (job profile
  `lp-ip.solve/1`);
- the objective is a black box, nonsmooth, noisy or strongly multimodal, or
  gradients are unavailable: use
  [fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust) (`fcmaes.retry/1`:
  BiteOpt, DE, CMA-ES, CR-FM-NES or PGPE retries);
- several objectives need a Pareto front, or the user wants good solutions
  across behaviour niches: fcmaes-rust's `fcmaes.mode/1` or
  `fcmaes.map-elites/1`;
- each evaluation is an expensive deterministic simulation without
  derivatives, with box bounds only, 1–32 variables and at most 256
  evaluations: the experimental `fcmaes.surrogate/1` (exact-GP Bayesian
  optimization; confirm it in the fcmaes worker's `describe` output);
- there are more than 256 variables, or exact sparse second derivatives are
  available and matter: a sparse interior-point or exact-Hessian SQP code
  (the IPOPT, WORHP, KNITRO family) fits better; SQP-craft has no
  exact-Hessian option. On CUTEst, exact Hessians raised Uno's filterSQP from
  760 to 845 solved problems ([benchmark](docs/benchmark-results.md));
- the application needs a global optimum, a second-order certificate or a
  certificate of infeasibility: none is provided;
- the variables are integer: there is no integer support (LP-IP-craft for
  linear MIPs; fcmaes-rust decodes integer variables in black-box search).

Their workers can be registered on the same `optimization-run` node and
reached through the same `optimization-mcp` server
([Agents over MCP](#agents-over-mcp)).

For a smooth but multimodal problem, start with SQP multistart
(`batch::solve_many`, or several `starts` in one job). fcmaes-rust
documents a global layer around a local solver: its retries find promising
basins whose points become SQP starts, or an external solver runs inside the
closure of its `retry`
([interoperation pattern](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/optimizer-boundary.md#supported-interoperation-pattern)).
Its docs advise adding local refinement only after a paired comparison shows
it beats spending the same budget on the global method. The job layer has no
cross-solver pipeline: the agent passes points between jobs itself, which
needs the same model registered in both workers.

## Choose the integration surface

| Need | Entry point | Scope |
|---|---|---|
| Rust model | `NativeSqp<B>` with `ClarabelQp` or `ActiveSetQp` | Full option set, observer, cancellation |
| Python model | `sqp_craft.solve(..., method="sqp")` | Same solver; callbacks run on the calling thread |
| Many starts, Rust model | `sqp_craft_solver::batch::solve_many` / `try_solve_many` | Ordered results, worker-local models and backends |
| Many starts, Python model | Separate processes (the spec is picklable), `examples/multistart_processes.py` | Python callbacks cannot run on solver threads |
| Durable, CPU-budgeted agent experiments | `optimization-run` or `optimization-mcp` + `sqp-job-worker` or an application worker | Registered native models: the benchmarks or your own; see below |
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
- Keep the box tight and meaningful: a solve never evaluates the model
  outside it. `sqp.evaluate/1` evaluates given points as they are, outside
  the box too, so return an error where the model is undefined.
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
the four performance fixtures of [docs/active-set-qp.md](docs/active-set-qp.md)
(orbit transfers with 84 and 164 variables, a separable and a restoration
case), the active-set backend takes the same steps as Clarabel and makes the
whole solve 5.4–6.4× faster; across CUTEst it is about 4.5% faster in time and
2% cheaper in callbacks.

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
values, and pass the solver's multipliers. The job worker does the
equivalent with `kkt::residuals` on a `KktPoint`.

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
  `examples/multistart_processes.py` runs distinct starts with a
  `ProcessPoolExecutor`, one native thread per process. The model libraries'
  pools (`OMP_NUM_THREADS` and friends) are set to `--model-threads`
  (default 1), overwriting inherited values. The worker count stays within
  the usable CPUs: the affinity mask lowered to the cgroup quota, as `node
  doctor` counts them. The best result is rechecked with `kkt_residuals`.
- Native models may implement `NlpProblem::set_threads` to split their own
  propagation with `sqp_craft_parallel::Execution` within the declared budget.
- `CancellationToken` / `Cancellation` is checked before every model call and
  inside the QP backend; it cannot interrupt a running callback. Ctrl-C ends a
  Python solve with `KeyboardInterrupt`. An observer receives `Progress`
  (iteration, objective, primal violation) and can stop the solve, which ends
  `cancelled`.

### Many cores

- In process, `batch::solve_many` allows `workers * options.threads <= 8`
  (outer times inner).
- The job layer schedules more outer tasks: one job with N distinct starts
  and `outer_workers = N`, `inner_threads = 1` (`cpu_slots = N`, up to the
  node's capacity and 1024) runs N solves at once, each with its own model
  instance and QP backend.
- Inner threads rarely help: Clarabel's engine is serial, and the active-set
  backend splits work only from 512 variables (its dense kernels; an NLP has
  at most 256) or 32,768 Jacobian entries in bounded rows (row norms, with at
  least 8 rows; `crates/activeset/src/lib.rs`).
- So the first many-core experiment is many distinct starts with
  `inner_threads = 1`; try `inner_threads > 1` only for a native model whose
  own `set_threads` propagation is expensive.

## Shared agent job API

Start with the shared guide in
[optimization-jobs](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md)
([docs/job-api.md](docs/job-api.md) maps its sections) and
[the SQP tutorial](tutorials/job-api/README.md). The job layer gives an agent
durable, validated, CPU-admitted multistart solves with journals,
cancellation and result manifests. Its profile is deliberately smaller than
the in-process API.

```sh
cargo build --release --locked -p optimization-runner -p sqp-job-worker
cargo build --release --locked -p sqp-job-worker --example custom_worker
python3 tutorials/job-api/run.py --root /tmp/sqp-job-tutorial
./target/release/optimization-run --root /tmp/sqp-job-tutorial \
  problems describe sqp
```

The tutorial creates an eight-slot node, registers the stock worker as `sqp`
and the application worker `examples/custom_worker` as `sqp-curve-fit`, and
runs the simplex-constrained quadratic (objective 0.25 at `x = 0.25`) and
Rosenbrock with the active-set backend and a shifted quadratic with Clarabel,
eight starts each, evaluates three points with `sqp.evaluate/1`, fits a decay
curve to a hashed CSV in elastic mode and rechecks the fit in Python. Then it
runs every example of both workers' `describe` and checks every payload
against the `payload_schemas` (with `jsonschema`). It takes the binaries
from `target/release` (or `$CARGO_TARGET_DIR/release`; `--bin-dir` overrides
both). Commands for a single job are in the tutorial.

### Discovery: schemas and examples

The worker's `describe` (`optimization-run problems describe ID`, MCP
`describe_worker`) lists JSON Schemas of each payload (`payload_schemas`,
draft 2020-12, strict, with per-model rules for version, dimensions and
config; cross-field rules in `$comment`) and runnable examples (`examples`:
one job per model and profile, on one CPU slot, in resolved form, with any
input file as `content_text`). A client without a checkout builds a job from
an example by adding `schema_version`, a `request_key`, the worker ID and
`payload_schema` (= `profile`), and by writing the inputs and filling in
their `sha256`. Each model entry of `problems` has `id`, `version`,
`description`, `dimension`, `profiles`, `bounds`, and `config` (its JSON
Schema) and `inputs` where it has them. The MCP tool `put_input` stores a
file in the node's content-addressed input store: an MCP-only client stages
the CSV of an application model there, names the returned `file` and
`sha256` in the job's `inputs`, and validates and submits the job with the
returned `input_dir`.

### Registered models and application workers

Jobs cannot carry Python or other callbacks. A job names a model of the
worker's registry; the stock `sqp-job-worker` registers three benchmarks,
version 1 each, which read no inputs and have no config:

| Problem ID | Model | Bounds and dimensions |
|---|---|---|
| `benchmark.quadratic` | Sum of `(x[i]-(i+1)/n)^2` | `[-5,5]`, 1–256 variables |
| `benchmark.simplex` | Sum of `x[i]^2`, subject to `sum(x) = 1` | `[0,1]`, 1–256 variables |
| `benchmark.rosenbrock` | Adjacent-pair Rosenbrock | `[-5,5]`, 2–256 variables |

A user's model runs in an *application worker*: a small binary that depends
on the `sqp-job-worker` library, registers models in a
`sqp_job_worker::Registry` and runs `Adapter::new(registry)` with
`optimization_job_api::worker::run` ([models guide](docs/job-api/models.md)).
Each model has an id, a version and a description (`ModelInfo`), a typed
`config` with a JSON Schema, declared data inputs, and a factory that builds
the `ProblemSpec` and the `NlpProblem` for the job's dimension, config and
input bytes (already SHA-256 checked). The factory returns a `Model`
(`Model::new(spec, prototype)` clones the prototype; `Model::from_fn`), and
every task and every recheck gets a fresh instance, so parallel tasks share
no model state. `examples/custom_worker.rs` is a complete worker: a
constrained least-squares fit to a CSV input, with analytic derivatives and
`config.anchor_time`. Before advertising a model, call `registry.check()`
in a test: it compares the gradient and the Jacobian with finite differences
at the example start and the declared check points, so a wrong derivative
fails before any job (`custom_worker check` runs it from the command line).
Never deserialize executable callbacks.

### Job specification

`payload_schema` is `sqp.solve/1`. Unknown fields are rejected. Required:

| Payload field | Meaning |
|---|---|
| `problem`, `dimension` | Registered problem ID and its dimension |
| `starts` | One complete finite start per task, inside the model box (never projected) |
| `backend` | `active_set` or `clarabel` (with an underscore, unlike Python's `"active-set"`) |
| `max_iterations` | Accepted steps per task; 0 permits only the initial evaluation |
| `max_callback_calls` | Positive per-task function plus derivative calls |
| `feasibility_tolerance`, `stationarity_tolerance` | Positive finite values |

Optional:

| Payload field | Meaning |
|---|---|
| `problem_version` | Must equal the registered version (default) |
| `config` | The model's parameters (`problems[].config` schema); default: the model's defaults; `{}` for a model without parameters |
| `elastic_mode` | `true`: ℓ1-elastic step QPs instead of restoration; needs `backend: "active_set"` (rejected at validation with Clarabel); default `false` |

The envelope's `inputs` must be exactly the inputs the model declares
(`problems[].inputs`), none for the benchmarks. The resolved job records
`problem_version`, `elastic_mode` and, for a model with parameters, the
effective `config`. The CUTEst results recommend `active_set` with
`elastic_mode: true` ([Options](#options)). All other `SqpOptions` keep their
defaults. The remaining wall time is each task's cooperative time limit,
and `inner_threads` (1–8) maps to `Options::threads`. For the benchmarks
`inner_threads` above 1 changes nothing; use `8 / 8 / 1`
(`cpu_slots / outer_workers / inner_threads`) for multistart throughput
([Many cores](#many-cores)).
The runner requires `outer_workers * inner_threads <= cpu_slots <= 1024`.
Validation echoes the specification and its output is capped at 4 MiB, so
about 600,000 one-dimensional starts fit, not the 1,000,000 tasks the
envelope allows.

### Job results

Inspect the job `state` first: `completed` only means every task returned.
Then read each outcome with `optimization-run result JOB_DIRECTORY`
(schema `optimization.result/0.2`):

| Outcome field | SQP meaning |
|---|---|
| `termination` | The native status |
| `claim` | `first_order` (native `converged` plus the worker's own KKT recheck), `feasible` (primal feasible, no first-order claim), `locally_infeasible`, `infeasible`, or `none` |
| `evidence` | `level` `rechecked` with `checks` `objective`, `bounds`, `constraints` and, with multipliers, `kkt`; `level` `claimed` for `locally_infeasible`, `infeasible` and an unrefuted `none` (`checks` lists what was still recomputed); `refuted` (`claim`, `verdict`, `message`) when the recheck rejected the native claim |
| `x`, `feasible`, `objective` | `x` is the solver's candidate; `feasible` and `objective` come from a fresh model evaluation at it |
| `details` | `kkt` residuals, `primal_violation`, `qp_backend`, `qp_status`, `message`, `global_optimum_certified: false`, the `acceptance` thresholds, the effective `options` (`backend`, `elastic_mode`, budgets, `threads`) and, in elastic mode, `elastic` (steps, final weight, weight increases) |

The recheck uses `feasibility_tolerance`, `stationarity_tolerance`,
complementarity ≤ 1e-6 and dual sign ≤ 1e-8. A native `converged` that fails
it is not reported as `first_order`: `evidence.refuted` names `first_order`
with verdict `false_optimality` (the point is feasible, but the residuals fail
the acceptance or overflow; claim `feasible`), `infeasible_point` (claim
`none`; a point outside the variable bounds is judged so without evaluating
the model there, with `checks: ["bounds"]`) or `objective_mismatch` (the
model has no finite values at the point; claim `none`). A candidate or
multipliers of the wrong length or with non-finite entries are solver
defects and fail the task, as the shared contract states. A `converged` without multipliers is
neither confirmed nor refuted: claim `feasible` or `none` from the point, and
`details.note` says the first-order conditions were not checked.
`locally_infeasible` (elastic mode at a stationary point of the violation) and
`infeasible` (linear rows and bounds without a common point, as the QP solver
judged) map the native statuses with `claimed` evidence: the point's
infeasibility is recomputed, its stationarity is not. A point that turns out
feasible refutes the claim (`false_infeasibility`). `locally_infeasible`
occurs only with `elastic_mode`; the benchmarks are feasible, so neither
claim occurs with them.
Treat any refutation as a defect report. Work counters are `functions`,
`derivatives`, `iterations`, `qp_iterations` and `validation_calls`; never add
them to another profile's counters. Graceful cancellation sets the native
token. Killed workers leave `work` null and only committed tasks count; there
is no BFGS or solver checkpoint resume.

### Evaluate given points

`sqp.evaluate/1` evaluates points against a registered model without solving,
for example an agent's candidates or another solver's answer. Payload:
`problem`, `dimension`, `points` (finite vectors of that dimension, one task
each; they may lie outside the bounds, and the model is evaluated there,
so it must signal undefined points with an error, which gives claim `none`)
and `feasibility_tolerance`, and optionally `problem_version` and `config`
(absent or `null`: the registered defaults); inputs as for `sqp.solve/1`;
`inner_threads` must be 1. Points are inline: the resolved job must stay below the
runner's 4 MiB validation output. Each outcome has `termination:
"evaluated"`, claim `feasible` within the tolerance or `none`, the objective,
the constraint values and `primal_violation` in details, and `rechecked`
evidence. Only function values are computed. A point whose values overflow
gets claim `none`, no objective, unknown feasibility, `claimed` evidence and
the reason in `details.message`. The work counter is `evaluations`. See
`tutorials/job-api/simplex-evaluate.json`.

### Agents over MCP

Agents can drive the same jobs through `optimization-mcp`, the MCP server
(stdio, one node) that the runner's build also makes: thirteen tools
(describe, validate, submit, wait, result summaries, cancel, kill, repair)
over `optimization-run` and, with the MCP Tasks extension, a task per
submitted job that completes with its result. The node's owner initializes
it and registers the workers with the CLI first; then connect with absolute
paths:

```sh
claude mcp add sqp-jobs -- "$PWD/target/release/optimization-mcp" --root /abs/node
```

The tools, options and error codes are in
[Agents and MCP](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#agents-and-mcp).
The server registers no workers and never changes the node. `get_result`
returns a summary per task (claim, termination, objective, evidence level,
refutation); the points and `details`, with the KKT residuals, are in the
complete records it returns with `tasks`. `fcmaes-job-worker` and
`lp-ip-job-worker`, built in their own repositories, can be registered on the
same node; one server then reaches all three, and `describe_node` lists them.
Claims differ between workers (`first_order` here, `numerical_optimum` for
LP/MIP): do not equate them.

### Shared crates

The shared crates, the shared guide, the JSON Schemas, the integration
harness and the runner regression tests (`scripts/job-api/test_review.py`
with `fake_worker.py`) live in
[optimization-jobs](https://github.com/dietmarwo/optimization-jobs). This
repository depends on them at the git revision its root `Cargo.toml` pins;
`cargo build -p optimization-runner` still builds `optimization-run` and
`optimization-mcp` here,
and `scripts/job-api/shared_root.py` prints the path of Cargo's checkout, from
which CI runs the harness. Until October 2026 the three repositories carried
byte-identical copies. Version 0.1.1 synchronized them, after they had
drifted following the 0.1.0 delivery. It carries this repository's review fixes
([docs/job-api/REVIEW.md](docs/job-api/REVIEW.md)) to the other two:

- An unreadable journal seals the job `failed` with `runner.journal_error`
  instead of breaking `jobs list`, and appends end a torn journal tail first.
- A job whose launcher died while it was queued is repaired as `interrupted`,
  with zero work and complete accounting (fcmaes-rust's accounting combined
  with this repository's rule that detached attempts with a unit stay
  unknown).
- Staging uses a `.staging-*` directory under the node root, so `TMPDIR` may be
  `noexec` but the root must allow execution; a signal during validation stops
  the validator.
- Finished jobs are not flagged after a reboot, `./model.json` is a valid
  input path, and detached units are named `optimization-job-ID.service`.
- The result contract also knows the fcmaes `non_dominated_set` and
  `elite_archive` claims; SQP results never use them.

Version 0.2.0 adds the result contract `optimization.result/0.2` and worker
protocol `optimization.worker/0.2`: the claims `infeasible`, `unbounded`,
`infeasible_or_unbounded`, `locally_infeasible` and `certified_optimum`;
`evidence` with checks, refutations and a certificate slot;
`objective_bounds`; coded runner errors; and JSON Schemas
(`docs/job-api/schemas/` in optimization-jobs). The job envelope stays `optimization.job/0.1`. A 0.2
runner still accepts 0.1 workers, and a 0.1 runner rejects 0.2 workers at
registration. `describe` reports the shared version as
`build.job_api_version`. Run a node root through one runner build. The raw validation logs of the 0.1.0 delivery are under
`validation/job-api/`.

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

The main CI commands, from the repository root (`scripts/check.sh` runs the
main Rust and Python checks together):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo run --locked -p sqp-craft --example toy
cargo run --locked -p sqp-craft-solver --example solve
cargo test --release --locked -p sqp-job-worker
cargo build --release --locked -p optimization-runner -p sqp-job-worker
cargo build --release --locked -p sqp-job-worker --example custom_worker
SHARED=$(python3 scripts/job-api/shared_root.py)
export JOB_API_TARGET="$PWD/target/release" JOB_API_TUTORIALS="$PWD/tutorials/job-api"
python3 "$SHARED/scripts/job-api/test_integration.py"
python3 "$SHARED/scripts/job-api/test_review.py"
python3 "$SHARED/scripts/job-api/test_mcp.py"
python3 tutorials/job-api/run.py --root /tmp/job-api-tutorial
python -m pip install ".[test]" && python -m pytest -q
python examples/demo.py
python examples/solve_sqp.py
python examples/restore_hs071.py
python examples/orbit_transfer.py 20
python examples/multistart_processes.py
```

CI also checks the solver crates with Rust 1.85 (excluding the job worker) and
the job worker with Rust 1.88, and runs the SLSQP comparison and the random, complementarity and SNOPT
stress families, which fail on any false convergence claim, a 20-problem
CUTEst subset, and a build without the native solver. Pass `--fcmaes` and
`--lp-ip` executables to `test_integration.py` to test a mixed node;
optimization-jobs' CI does that with all three workers.

## Repository references

- [README.md](README.md): release notes, Python usage, build and tests
- [docs/api.md](docs/api.md): the contract: problem, callbacks, options, statuses, errors, KKT, QP seam, batches
- [docs/benchmark-results.md](docs/benchmark-results.md): CUTEst results and failure classes
- [docs/gaps.md](docs/gaps.md): weaknesses compared with mature SQP codes
- [docs/active-set-qp.md](docs/active-set-qp.md), [docs/elastic-mode.md](docs/elastic-mode.md),
  [docs/second-order-correction.md](docs/second-order-correction.md),
  [docs/snopt-lessons.md](docs/snopt-lessons.md): the opt-in mechanisms
- [docs/performance-60/REPRODUCE.md](docs/performance-60/REPRODUCE.md): thread budgets and batches
- [docs/job-api.md](docs/job-api.md) (where the shared guide lives),
  [docs/job-api/models.md](docs/job-api/models.md) (application models and workers),
  [tutorials/job-api](tutorials/job-api/README.md),
  [docs/job-api/REVIEW.md](docs/job-api/REVIEW.md),
  [docs/job-api/VALIDATION.md](docs/job-api/VALIDATION.md): shared job API and this worker
- [optimization-jobs guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md):
  envelope, runner, result contract, MCP server
- `ai-context.md` of [fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust/blob/main/ai-context.md)
  and [LP-IP-craft](https://github.com/dietmarwo/LP-IP-craft/blob/main/ai-context.md):
  the sibling solvers
- [docs/target-card.md](docs/target-card.md): the open application contract
- `crates/core/src/kkt.rs`, `crates/core/src/derivative_check.rs`: independent checks
- `crates/sqp-job-worker/src/lib.rs` (adapter), `registry.rs` (`Registry`, `ModelInfo`, `Model`), `benchmarks.rs`,
  `solve.rs` and `evaluate.rs` (the two profiles), `discovery.rs` (payload schemas and examples), and
  `examples/custom_worker.rs` (an application worker)
- [AGENTS.md](AGENTS.md): build, test and conventions for coding agents
- `python/sqp_craft/__init__.py`: Python API and defaults
