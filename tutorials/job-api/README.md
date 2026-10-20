# Tutorial: native SQP jobs and multistart batches

This tutorial runs three registered smooth models: a simplex-constrained
quadratic, Rosenbrock, and an unconstrained shifted quadratic using Clarabel.
Each job has eight independent initial points and an eight-slot CPU budget.
Models and QP state are private to each task; no Python callback is serialized.
A fourth example evaluates given points without solving.

## Build and execute

From the repository root, on Linux with Rust 1.88+ and Python 3:

```sh
cargo build --release --locked -p optimization-runner -p sqp-job-worker
python3 tutorials/job-api/run.py --root /tmp/sqp-job-tutorial
```

The script takes the binaries from `target/release`, or from
`$CARGO_TARGET_DIR/release` when that is set; `--bin-dir` overrides both. It
initializes a node, registers the worker, validates the examples and
runs them. It independently asserts the expected objective, the `first_order`
claim and its evidence, and the evaluated points, prints attempt directories,
and saves `tutorial-report.json` under the node. If the `jsonschema` package is
installed, it also validates every job and result against the published
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

The agent sees the worker with `describe_worker` (worker `sqp`), can pass
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

`max_iterations` limits accepted steps (zero permits only initial evaluation);
`max_callback_calls` limits function plus derivative calls and must be positive.
The job wall-time limit applies to the batch after admission and is passed as
remaining cooperative time to each solve. `feasibility_tolerance` and
`stationarity_tolerance` are positive finite values. Independent acceptance also
uses complementarity <=1e-6 and dual-sign residual <=1e-8 in original units.
Other native SQP options are not exposed by this initial profile.

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
evidence; this profile keeps elastic mode off, so they do not occur with the
registered models.

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

## Eight-slot layouts and custom native models

Use `cpu_slots=8, outer_workers=8, inner_threads=1` for multistart throughput, or
`8 / 1 / 8` for one solve with an inner numerical allowance. Small kernels and
full BFGS may remain serial; Clarabel's engine is serial. The registered model
functions here are inexpensive and serial, and the active-set QP kernels split
their work only from 512 variables or 32,768 bounded-row entries, which these
models (at most 256 variables and one row) never reach. For them `inner_threads`
above 1 changes nothing; batch parallelism is the useful choice. `8 / 8 / 8` is rejected.

To add a trajectory model, extend `registered` and the model implementation in
`crates/sqp-job-worker/src/main.rs`, or create a separate worker using the shared
SDK. Implement `NlpProblem::functions` and `derivatives`, with a validated
`ProblemSpec` and analytic Jacobian pattern. Construct mutable model/backend
instances inside each task. Add a versioned, strictly typed configuration and
optional hashed data inputs if needed; never deserialize executable callbacks.
Use `set_threads` only if the model honors the declared inner budget without
creating another independent full-size pool. Run derivative finite-difference
checks and known numerical acceptance tests before advertising the model.

The three model derivatives already have such a test in this worker crate.
Run `cargo test --release --locked -p sqp-job-worker` and the shared
acceptance harness ([shared job API](../../docs/job-api.md#running-the-shared-tests-here))
for its acceptance tests. See the
[shared API guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md)
for cancellation, lifecycle, mixed-worker nodes and recovery; see
[native API](../../docs/api.md) for the larger in-process solver surface.
