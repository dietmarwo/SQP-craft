# Tutorial: native SQP jobs and multistart batches

**Prompt-driven adaptation:** [agent walkthrough](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic/sqp) using this repository's existing worker.

This tutorial runs three registered smooth models: a simplex-constrained
quadratic, Rosenbrock, and an unconstrained shifted quadratic using Clarabel.
Each job has eight independent initial points and an eight-slot CPU budget.
Models and QP state are private to each task; no Python callback is serialized.
A fourth example evaluates given points without solving. A fifth runs an
application model in a worker of its own: a constrained curve fit to a
hashed CSV file, in elastic mode.

## Build and execute

From the repository root, on Linux with Rust 1.88+ and Python 3:

```sh
cargo build --release --locked -p optimization-runner -p sqp-job-worker
cargo build --release --locked -p sqp-job-worker --example custom_worker
python3 tutorials/job-api/run.py --root /tmp/sqp-job-tutorial
```

The script takes the binaries from `target/release`, or from
`$CARGO_TARGET_DIR/release` when that is set; `--bin-dir` overrides both. It
initializes a node, registers the stock worker as `sqp` and the application
worker as `sqp-curve-fit`, validates the examples and
runs them. It independently asserts the expected objective, the `first_order`
claim and its evidence, the evaluated points and the curve fit, prints
attempt directories, and saves `tutorial-report.json` under the node. Then
it turns every example of both workers' `describe` into a job and runs it,
as a client without a checkout would. If the `jsonschema` package
(4.18 or newer) is installed, it checks every tutorial and example payload
against the worker's `payload_schemas`, checks that the schemas reject a few
invalid payloads the worker also rejects, and validates every job and result
against the published
[job](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api/schemas/optimization.job-0.1.schema.json) and
[result](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api/schemas/optimization.result-0.2.schema.json)
schemas, from Cargo's checkout of optimization-jobs at the pinned revision. The simplex example reaches
`x=[0.25,0.25,0.25,0.25]`, objective 0.25.
Rosenbrock and the shifted quadratic approach objective zero.

## One job, step by step

```sh
./target/release/optimization-run --root /tmp/sqp-manual node init --capacity 8
./target/release/optimization-run --root /tmp/sqp-manual workers add \
  --id sqp --executable ./target/release/sqp-job-worker
./target/release/optimization-run --root /tmp/sqp-manual problems describe sqp
./target/release/optimization-run --root /tmp/sqp-manual validate \
  --spec tutorials/job-api/simplex.json
./target/release/optimization-run --root /tmp/sqp-manual run \
  --spec tutorials/job-api/simplex.json
```

Read `run.json` from the returned directory, or use `optimization-run result
JOB_DIRECTORY`. `work` separates function calls, derivative calls, accepted
iterations, QP iterations and independent validation calls.

## Through an agent (MCP)

The runner's build also makes `optimization-mcp`, an MCP server over the same
node. With the node of the previous section, connect an MCP client, for
example Claude Code:

```sh
claude mcp add sqp-jobs -- "$PWD/target/release/optimization-mcp" --root /tmp/sqp-manual
```

The agent sees the worker with `describe_worker` (worker `sqp`), including
its payload schemas and runnable examples, can pass
any job of this directory to `submit_job` as `spec_path` (its input files are
read next to it), follows it with `wait_job` and reads each task's claim and
evidence with `get_result` (the KKT residuals are in the complete records that
`get_result` returns for `tasks`); `cancel_job` stops it gracefully. With
the MCP Tasks extension, `submit_job` returns a task that completes with the
job's result. See [Agents and MCP](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#agents-and-mcp).

## Registered problems and options

| Problem ID | Model | Bounds and dimensions |
| --- | --- | --- |
| `benchmark.quadratic` | Sum of `(x[i]-(i+1)/n)^2` | `[-5,5]`, 1–256 variables |
| `benchmark.simplex` | Sum of `x[i]^2`, subject to sum(x)=1 | `[0,1]`, 1–256 variables |
| `benchmark.rosenbrock` | Standard adjacent-pair Rosenbrock objective | `[-5,5]`, 2–256 variables |

Copy a JSON example and choose a new `request_key` for a new experiment.
`dimension` controls the registered model. `starts` contains one complete vector
per task; every vector must be finite, have that dimension and lie in the model
box. A start may violate the nonlinear/row constraints. It is never projected
silently. `backend` is `active_set` or `clarabel`; both are built into this worker.
The worker's `describe` lists a JSON Schema of each payload (`payload_schemas`)
and runnable examples (`examples`).

`max_iterations` limits accepted steps (zero permits only initial evaluation);
`max_callback_calls` limits function plus derivative calls and must be positive.
The job wall-time limit applies to the batch after admission and is passed as
remaining cooperative time to each solve. `feasibility_tolerance` and
`stationarity_tolerance` are positive finite values. Independent acceptance also
uses complementarity <=1e-6 and dual-sign residual <=1e-8 in original units.
Optional fields: `elastic_mode` (default `false`) solves every step QP as an
l1-elastic QP instead of entering feasibility restoration; it needs
`backend: "active_set"`, is rejected at validation with Clarabel, and appears
with the other effective options in each outcome's `details.options`.
`problem_version` must equal the registered version (the default), and
`config` holds a model's parameters (`{}` or omitted for the benchmarks).
The resolved job records the version, `elastic_mode` and the effective
config. Other native SQP options keep their defaults.

## Interpret convergence correctly

A lifecycle `completed` says all tasks returned; it does not assert convergence.
Each task includes the native termination, candidate feasibility and a claim.
`first_order` requires native `converged` status plus independently recomputed
objective, gradient, constraint/Jacobian values and KKT residuals. It is neither
a global optimum certificate nor a second-order sufficiency claim. Local
restoration failure or `locally_infeasible` is not a global infeasibility proof.

Each outcome's `evidence` says what was verified: `rechecked` with `checks`
`objective`, `bounds`, `constraints` and `kkt` for a `first_order` claim. If
the recheck rejects a native `converged`, the claim is not reported:
`evidence.refuted` names `first_order` with verdict `false_optimality` (claim
`feasible`; also when the residuals overflow), `infeasible_point` (claim
`none`; a point outside the variable bounds is not evaluated) or
`objective_mismatch` (no finite model values at the point; claim `none`).
Report such a refutation as a defect. A `converged` without multipliers is neither confirmed nor refuted;
`details.note` says so. `locally_infeasible` and `infeasible` map the native
statuses of elastic mode and of infeasible linear rows, with `claimed`
evidence; the benchmarks are feasible, so they do not occur with them.

A limited task may retain a feasible point with claim `feasible`. If a process
is killed, uncommitted active work is unknown and the common result has
`work:null`; completed task records survive. There is no BFGS, active-set or
solver checkpoint resume in this job protocol.

## Evaluate given points

`simplex-evaluate.json` uses `payload_schema: "sqp.evaluate/1"` to check three
points against the simplex model without solving:

```json
"payload": {"problem": "benchmark.simplex", "dimension": 4,
            "points": [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]],
            "feasibility_tolerance": 1e-07}
```

Each point is one task with `termination: "evaluated"`: the first two satisfy
`sum(x) = 1` and are `feasible` (objectives 0.25 and 0.3), the third is `none`
with `primal_violation` 1. Points may lie outside the bounds; they are then
infeasible. A point whose values overflow is `none` without an objective, with
the reason in `details.message`. Points are inline: keep the job well below
the runner's 4 MiB validation output. Use `inner_threads` 1.

## An application model in a worker of its own

`crates/sqp-job-worker/examples/custom_worker.rs` is a complete worker built
on the `sqp-job-worker` library: the benchmarks plus `app.decay-fit`, a
least-squares fit of `y = a*exp(-b*t) + c` to the rows of a CSV input,
constrained to pass through the row at `t = config.anchor_time`, with
analytic derivatives. `decay-fit.json` runs it on `decay.csv` from four
starts in elastic mode:

```json
"inputs": [{"name": "data", "file": "decay.csv", "sha256": "b779f43d..."}],
"payload": {"problem": "app.decay-fit", "problem_version": "1", "dimension": 3,
            "config": {"anchor_time": 1.0}, "starts": [[5.0, 2.5, 0.0], ...],
            "backend": "active_set", "elastic_mode": true, ...}
```

```sh
./target/release/optimization-run --root /tmp/sqp-manual workers add \
  --id sqp-curve-fit --executable ./target/release/examples/custom_worker
./target/release/optimization-run --root /tmp/sqp-manual run \
  --spec tutorials/job-api/decay-fit.json
```

The runner checks the CSV's SHA-256 and copies it into the attempt; the
model declares the input `data`, so the job must name exactly that input
(the benchmarks reject any). Every start reaches the same `first_order`
point near `a = 3`, `b = 0.7`, `c = 0.5`, the parameters the data was made
with. `run.py` rechecks it from the CSV: the loss, the anchor row and
stationarity (the loss gradient is a multiple of the row gradient, since no
bound is active). [Application models](../../docs/job-api/models.md)
explains the registry.

## Eight-slot layouts and custom native models

Use `cpu_slots=8, outer_workers=8, inner_threads=1` for multistart throughput, or
`8 / 1 / 8` for one solve with an inner numerical allowance. Small kernels and
full BFGS may remain serial; Clarabel's engine is serial. The registered model
functions here are inexpensive and serial, and the active-set QP kernels split
their work only from 512 variables or 32,768 bounded-row entries, which these
models (at most 256 variables and one row) never reach. For them `inner_threads`
above 1 changes nothing; batch parallelism is the useful choice. `8 / 8 / 8` is rejected.

To add a trajectory model, register it in an application worker like
`examples/custom_worker.rs` ([Application models](../../docs/job-api/models.md)).
Implement `NlpProblem::functions` and `derivatives`, with a validated
`ProblemSpec` and analytic Jacobian pattern. The worker makes a fresh model
instance and QP backend for each task. Give the model a version, a typed
config and hashed data inputs if needed; never deserialize executable
callbacks. Use `set_threads` only if the model honors the declared inner
budget without creating another independent full-size pool. Call
`registry.check()` in a test (derivatives against finite differences) and
run known numerical acceptance tests before advertising the model.

The benchmarks and the example model have such tests in this worker crate.
Run `cargo test --release --locked -p sqp-job-worker` and the shared
acceptance harness ([shared job API](../../docs/job-api.md#running-the-shared-tests-here))
for its acceptance tests. See the
[shared API guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md)
for cancellation, lifecycle, mixed-worker nodes and recovery; see
[native API](../../docs/api.md) for the larger in-process solver surface.
