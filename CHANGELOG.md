# Changelog

All notable changes to SQP-craft are documented here. Versions are the
workspace version in Cargo.toml (Rust crates, Python package sqp-craft and
the job worker share it). No version has been published to a registry yet;
dates are those of the commits that set each version.

## [Unreleased]

### Added

- The [prompt-driven SQP tutorial](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic/sqp) solves a simplex allocation with both QP backends from four starts and runs Rosenbrock multistart through the existing `sqp-job-worker`, rechecking objectives and stationarity. No numerical code or dependency pin changes.

- The [prompt-driven curve-fitting tutorial](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic/curve-fit) runs the existing constrained decay-fit worker and checks its half-sum-of-squares objective, anchor equality and first-order evidence.

### Changed

- Move the optimization-jobs pin to `466a373` (source after 0.3.0):
  opt-in trusted command jobs and verified binary staging/collection in the
  shared runner. Command/file operations use the CLI; the optimization SDK,
  numerical algorithms, envelopes, worker protocols and result claims are unchanged.

- optimization-jobs pin `70e210f` (after tag `v0.3.0`; the crates are still
  0.3.0) in the manifest and the lockfile. The runner gains `node
  capabilities` and `jobs execution JOB_ID`, an execution record of how each
  job's worker process ended. Envelope, result contract, worker protocol and
  solver behavior are unchanged.
- README badges as in fcmaes-rust: pure-Rust solver, `unsafe` forbidden in the library crates, no C/C++
  dependencies, crates.io, docs.rs, PyPI and the AI agent guide.
- optimization-jobs pin `37b30e4` (tag `v0.3.0`, the first crates.io release
  of `optimization-job-api` and `optimization-runner`): crate metadata, the
  shared guide that `get_guide` serves names the October 2026 registry
  releases, and the job crates are 0.3.0. Envelope, result contract and
  worker protocol are unchanged.
- Clippy 1.88, the job worker's minimum Rust, is clean on the workspace too:
  the worker's thread-limit message inlines its format argument. The text is
  unchanged.

## [0.17.0] - 2026-10-22

The first release on PyPI (`sqp-craft`) and crates.io (`sqp-craft`,
`sqp-craft-parallel`, `sqp-craft-activeset`, `sqp-craft-clarabel`,
`sqp-craft-solver`). The first crates.io upload needs an API
token; later releases use Trusted Publishing ([RELEASING.md](RELEASING.md)).

Everything merged after the commit that set 0.16.0 (`8c9bc0c`, #11): #10
and #12 to #21, and the release preparation. The code of the solver crates
and of the Python package is unchanged since 0.16.0; the crates gained
publishing metadata.

### Added

- CUTEst benchmark harness `bench/` on the S2MPJ problems, with SciPy SLSQP
  and Uno's filterSQP and IPOPT presets as comparators and one independent
  first-order check per run; first results on 1,036 problems in
  `docs/benchmark-results.md`, tests in `tests/test_bench.py`, and a CI job
  on 20 Hock-Schittkowski problems that fails on a sqp-craft false claim
  (#10). Measured with 0.14: Clarabel solves 754 problems, the active-set
  backend 799 and 813 with elastic mode, with no false claim in 12,432
  sqp-craft runs. The 0.16 review reran it: the active-set backend solves
  801, Clarabel and elastic mode are unchanged.
- Shared agent job API 0.1 (#12): `optimization-job-api` (versioned
  envelope, input hashes, checked journals, result contract, worker SDK),
  `optimization-runner` with the runner `optimization-run` (worker registry,
  staging, idempotent request keys, CPU-slot admission, process-group
  supervision, cancellation, deadlines, journal recovery) and
  `sqp-job-worker` with the profile `sqp.solve/1` (registered models
  `benchmark.quadratic`, `benchmark.simplex` and `benchmark.rosenbrock`,
  active-set or Clarabel backend, independent KKT recheck). Guide, tutorial
  `tutorials/job-api/`, integration harness and
  `docs/job-api/VALIDATION.md`. The job layer needs Rust 1.88; CI checks
  the solver crates with Rust 1.85 and the job layer with 1.88.
- `ai-context.md`: operational context for AI agents, covering
  formulation, options and backend choice, statuses and multiplier signs,
  independent KKT checks, multistart and the job worker (#13).
- Result contract 0.2 (#15): shared job crates 0.2.0 with the result schema
  `optimization.result/0.2` (new claims; evidence with checks, refutations
  and a certificate slot; objective bounds), worker protocol 0.2 with 0.1
  still accepted, coded runner errors, `already_terminal` on cancelling a
  finished job, and JSON Schemas for the job envelope and the result
  manifest. The worker adds the point-evaluation profile `sqp.evaluate/1`.
- `optimization-mcp`, the MCP server over one node, from optimization-jobs
  `0ee495f`; `cargo build -p optimization-runner` builds it here, and CI
  runs the shared MCP tests (`test_mcp.py`) with this worker (#18).
- `tutorials/job-api/run.py --bin-dir`; the script also honors
  `CARGO_TARGET_DIR` and reports a missing binary clearly (review SQP-04,
  #19).
- Release workflows modeled on fcmaes-rust, with `RELEASING.md`:
  `python-release.yml` builds abi3 wheels (CPython 3.10+) for Linux x86_64,
  macOS x86_64 and arm64 and Windows x86_64 plus the source
  distribution, installs and tests each one, attests them and publishes
  `sqp-craft` to PyPI through Trusted Publishing on a `v*` tag that equals
  the workspace version (a manual run builds and tests without publishing).
  `publish-crates.yml` checks and publishes the five library crates
  (`sqp-craft`, `sqp-craft-parallel`, `sqp-craft-activeset`,
  `sqp-craft-clarabel`, `sqp-craft-solver`) to crates.io once the repository
  variable `CRATES_IO_TRUSTED_PUBLISHING` is `true`; until then it runs
  every check and skips the upload.
- The library crates are publishable: crates.io metadata, a README and the
  license in each crate, and versions on their path dependencies. The
  Python package, the worker and the benchmarks stay unpublished crates.
- README section "Install from source": requirements (Rust 1.85, 1.88 for
  the job layer, a C linker but no external optimizer, BLAS or CMake), the
  Python build, the job worker with its git dependency on optimization-jobs,
  and the crates as git dependencies.
- `ai-context.md` points to the AI entry point of optimization-jobs, which
  chooses between the four solver repositories, and says what the release
  workflows will publish.
- Application models for the job worker (agent-discovery review F03):
  `sqp-job-worker` is a library plus a thin binary. `sqp_job_worker::Registry`
  takes models with an id, a version and a description (`ModelInfo`), a
  typed `config` with its JSON Schema, declared data inputs (SHA-256 checked
  by the runner and the worker) and a factory that builds the `ProblemSpec`
  and a `Model`, which makes a fresh `NlpProblem` instance for every task
  and every recheck. `Registry::check` and `Model::check_derivatives`
  compare the derivatives with finite differences before a model is
  advertised. The stock binary registers the three benchmarks through this
  API. `examples/custom_worker.rs` is a standalone application worker with
  `app.decay-fit`, a constrained least-squares fit to a CSV input; the
  tutorial runs it (`decay-fit.json`, `decay.csv`) and rechecks the fit in
  Python, and `tests/custom_worker.rs` drives it through `prepare` and
  `solve`. Guide: `docs/job-api/models.md`.
- Optional payload fields of `sqp.solve/1` and `sqp.evaluate/1`:
  `problem_version` (must equal the registered version, the default) and
  `config` (the model's parameters); the envelope's `inputs` are accepted
  exactly when the model declares them. The benchmarks still reject inputs.
- `elastic_mode` in `sqp.solve/1` (review F10): optional, default `false`,
  needs `backend: "active_set"` and is rejected at validation with
  Clarabel. Outcomes report the effective options in `details.options` and,
  in elastic mode, the elastic counters in `details.elastic`.
- Discovery (review F01): the worker's `describe` lists `payload_schemas`
  (a strict JSON Schema per profile, with per-model rules for version,
  dimensions and config) and `examples` (one job per registered model and
  profile, in resolved form, inputs as `content_text`), built from the
  registry, so an application worker advertises its own models. Model
  entries add `version`, `description`, `profiles`, `config`, `inputs` and
  `bounds`. `tests/discovery.rs` turns every example of the stock worker,
  the custom worker and a worker without benchmarks into a job and runs it;
  the tutorial checks every example and tutorial payload against the
  schemas with `jsonschema>=4.18`. An explicit `null` for `problem_version`
  or `config` means the same as leaving it out, in the worker and in the
  schemas (`tests/discovery.rs` and the tutorial check absent, null, valid
  and invalid values for both profiles and both workers).
- `examples/multistart_processes.py`: many starts of a Python model in a
  `ProcessPoolExecutor`, one native thread per process, and a
  `kkt_residuals` recheck of the best result; CI and `scripts/check.sh` run
  it. The model libraries' thread pools are set to `--model-threads`
  (default 1), overwriting inherited values. The default worker count stays
  within the usable CPUs: the affinity mask lowered to the cgroup v1 or v2
  quota (`tests/test_multistart_example.py`). The script prints the
  topology.
- `AGENTS.md` for coding agents; `ai-context.md` gains a "Many cores"
  section, the discovery keys, the application registry (replacing the
  instruction to edit the worker's enum) and the MCP tool `put_input`.

### Changed

- crates.io and docs.rs pages: the core crate describes SQP with sparse
  (CSR) constraint Jacobians instead of "a sparse SQP solver" (the solver is
  dense BFGS), and its rustdoc drops the development note "Iteration 1"; the
  five crates' homepage is the SQP-craft entry of the ecosystem page; the
  READMEs show `cargo add`, and the solver README a complete, tested example
  with the limits; the solver's rustdoc links `docs/snopt-lessons.md`
  absolutely and no longer says "four" SNOPT mechanisms for a list of five.
- `publish-crates.yml`: while Trusted Publishing is not enabled
  (`CRATES_IO_TRUSTED_PUBLISHING`), a tag publishes the crates with the
  secret `CARGO_REGISTRY_TOKEN`, one crate at a time in dependency order and
  skipping versions already on crates.io. This publishes the first release
  without a local toolchain; `RELEASING.md` describes the token and its
  removal afterwards.
- The PyPI description is `python/README.md`, a short page with installation,
  an example, the limits and absolute links; the repository README, with
  its relative links and history, stays on GitHub.
- Discovery: the README opens with the title, a short description (dense
  BFGS, at most 256 variables, derivatives needed) and links to the shared
  solver-selection guide and the ecosystem overview; the five crate READMEs
  link them too. `pyproject.toml` adds keywords and project URLs
  (documentation, issues, changelog, ecosystem, agent guide). `RELEASING.md`
  covers discovery after a release.
- The README opens with a banner (`docs/images/sqp-craft.png`). It is
  linked by absolute URL, so the PyPI description shows it too.
- The resolved `sqp.solve/1` job records `problem_version` and
  `elastic_mode`, and the effective `config` of a model with parameters;
  the resolved `sqp.evaluate/1` job records `problem_version`. Every
  existing tutorial and harness job validates and gives the same outcomes
  and work counts; outcomes gain `details.options`.
- CI builds the `custom_worker` example for the tutorial.

- The job worker records a `converged` status that fails the recomputed
  KKT acceptance as a refutation instead of failing the job, maps
  `locally_infeasible` and `linear_infeasible` to claims with `claimed`
  evidence, and reports the workspace version as its engine version
  instead of a hard-coded string (#15).
- The shared job crates were synchronized as the 0.1.1 set of 20 files,
  identical in fcmaes-rust, LP-IP-craft and SQP-craft: zero-work repair of
  a queued job whose launcher died, the fcmaes set and archive claims, and
  shared `test_review.py` and `fake_worker.py`; CI checks the manifest
  (#14). The shared guide, plan and harness were synchronized with
  fcmaes-rust (#16).
- `optimization-job-api` and `optimization-runner`, the shared guide, plan,
  JSON Schemas and scripts moved to
  https://github.com/dietmarwo/optimization-jobs. The workspace depends on
  the crates at a pinned git revision instead of in-tree copies and their
  checksum manifest; `docs/job-api.md` is a pointer to the shared guide; CI
  and the tutorial run the shared harness and schemas from Cargo's checkout
  (`scripts/job-api/shared_root.py`) (#17).
- optimization-jobs pin: `9feb72e` (#17), `0ee495f` (#18), `eb1c9de` (#19:
  cancellation race, bound consistency in `Outcome::check`, MCP request
  limits) and `6a42472` (#20: worker-declared artifacts and paged artifact
  reads, `jobs artifact --offset/--limit` and MCP `get_artifact` with
  `offset` and `limit`; no change for this worker's profiles). The shared
  tests were rerun at each pin (`docs/job-api/VALIDATION.md`). Then
  `423cd87`: `artifact::page` streams the file and parses only the page's
  records; no change for this worker.
- optimization-jobs pin `069d87c` (its review of agent discovery):
  `optimization-mcp` gains `put_input` (stages an input file, such as a model
  or a CSV, and returns its `sha256` and `input_dir`) and `get_guide` (the
  cross-repository AI entry point); `node doctor` reports the host's usable
  CPUs and memory; `workers add` accepts up to 1 MiB of `describe` output;
  and the shared harness, which CI runs from Cargo's checkout, checks the
  worker's payload schemas and runs its examples, also over MCP.
- optimization-jobs pin `43ef3ae` (third review of agent workflows, N1;
  the merge of optimization-jobs #10, which adds only the landing page and
  discovery docs to `8d19278`):
  the guide that `optimization-mcp` serves with `get_guide` has the
  corrected Pareto guidance (scalarization gives candidates), and the
  shared harness, which CI runs from the pinned checkout, checks that the
  payload schemas and the worker agree on `null` optional fields. The new
  workflow `shared-pin.yml` fails when optimization-jobs `main` has changed
  the shared crates, guide, job API docs or test scripts since the pinned
  revision (weekly, and on changes to Cargo files), so the pin no longer
  drifts. Landing-page and discovery files do not count.
- `ai-context.md` reviewed against `main`: the MCP server, routing to the
  sibling workers LP-IP-craft and fcmaes-rust, and `inner_threads=1` as a
  requirement of the evaluate profile; the README names the sibling
  workers (#20).

### Fixed

- Job worker (review SQP-01): two finite but invalid `converged` results
  aborted the whole multistart job. Multipliers whose KKT residuals
  overflow now refute `first_order` with `false_optimality`; a finite
  candidate outside the variable bounds is judged infeasible
  (`infeasible_point`, checks `["bounds"]`) without evaluating the model
  there; a point without finite model values is refuted with
  `objective_mismatch`. Candidates or multipliers of the wrong length or
  with non-finite entries still fail the task. `validation_calls` counts
  the model calls actually made (#19).
- README layout table: the active-set backend depends on
  `sqp-craft-parallel` since 0.16, and `crates/parallel` and
  `crates/sqp-job-worker` were missing (#21).

## [0.16.0] - 2026-10-18

Set by #11 (`8c9bc0c`). It also contains #6 to #9, merged on 2026-10-17
while the workspace version was still 0.14.0 (their docs say "Version
0.14.0"). #11 squashes the reviewed 0.15 and 0.16 performance deliveries
(`docs/performance-30/`, `docs/performance-60/`); 0.15.0 has no commit of
its own on `main`.

### Added

- Dense dual active-set QP backend, crate `sqp-craft-activeset`
  (`ActiveSetQp`, #6): the Goldfarb-Idnani method on a dense Cholesky
  factor of H, which is also the exact positive-definiteness gate; a
  Householder reflector for entering constraints, Givens rotations for
  leaving ones; x and u recomputed from the factors with one refinement
  step before any result or infeasibility claim. Select it with
  `SqpOptions(qp_solver="active-set")` (`ValueError` for an unknown name)
  or `NativeSqp<ActiveSetQp>`; Clarabel stays the default. On the four
  fixtures it takes the same steps to the same objectives and the whole
  solve is 5.4-6.4x faster (orbit, 164 variables: 2,985 -> 546 ms); same
  status as Clarabel in all 4,500 paired stress runs.
- `SqpOptions.qp_warm_start` (default true): the next step QP starts from
  the last working set when the backend reports one. Clarabel reports
  none, so its runs are unchanged. A warm attempt that ends infeasible or
  fails the independent check is repeated cold (#6).
- `docs/gaps.md`: weaknesses compared with SNOPT, filterSQP, NLPQL, SLSQP
  and the IPOPT/WORHP/KNITRO family, and the order of work (#6).
- Elastic mode (#7): `SqpOptions.elastic_mode` (default false) and
  `elastic_weight` (default 1e4, times `max(1, |grad f(x0)|_inf)`) solve
  every step QP as an l1-elastic QP instead of entering restoration. It
  needs the active-set backend (`QpBackend::supports_elastic`; Clarabel
  rejects `QpOptions.elastic_weight`). Independent checks
  `QpModel::accepts_elastic` and `kkt::elastic_residuals_with_workspace`.
  New status `locally_infeasible` after six tenfold weight increases;
  report fields `elastic_steps`, `elastic_weight` and
  `elastic_weight_increases`. Stress family: 14 `locally_infeasible`
  against 15 restoration failures (13 the same); restoration stays the
  default.
- Second-order correction (#8): `SqpOptions.second_order_correction`
  (default false), Nocedal & Wright Algorithm 18.3 once per iteration, with
  both backends and in elastic mode; report fields
  `second_order_corrections`, `second_order_steps` and
  `IterationRecord.second_order_correction`. Maratos example from 200
  starts: -16% iterations, -21% callbacks; stress family: -2.3%
  iterations, two problems end in failed restoration instead.
- Four opt-in mechanisms from SNOPT (Gill, Murray and Saunders 2005;
  `docs/snopt-lessons.md`, #9); with all of them off the 9,000 stress runs
  are bit-identical:
  - `hessian_update="augmented"`: curvature repair with the
    augmented-Lagrangian term instead of damping; counters
    `bfgs_damped_updates`, `bfgs_modified_updates` and
    `IterationRecord.curvature`;
  - `merit="augmented-lagrangian"` and `violation_limit`, with
    `RunReport.violation_limit_rejections`;
  - declared linear rows, `ProblemSpec(..., linear_rows=[...])` or
    `ProblemSpec::with_linear_rows`: the start is projected onto them,
    every QP keeps them hard (`QpOptions.hard_rows`), new status
    `linear_infeasible`, `RunReport.linear_projection`;
  - `scaling="linear"` or `"jacobian"` (SNOPT's Scale options 1 and 2):
    the QP and the BFGS matrix are scaled by powers of two; report fields
    `variable_scales` and `row_scales`.
- Native thread budgets (#11): `Options.threads` is a maximum native budget
  of 1 to 8 (default 1; before, only 1 was accepted). New crate
  `sqp-craft-parallel` (bounded Rayon pools). The active-set backend splits
  large QP kernels (512 variables or 32,768 bounded-row entries)
  bit-identically to the serial path. Hooks `NlpProblem::set_threads` and
  `QpBackend::set_threads`, whose defaults keep the model serial and reject
  more than one backend thread. Python callbacks stay on the calling
  thread; the Clarabel engine stays serial.
- Batch solves (#11): `batch::solve_many` and `try_solve_many` solve
  ordered independent starts with worker-local models and backends,
  shared cancellation and a total budget of at most 8; outer tasks run on
  dedicated threads. Delivery host: 16 orbit-20 solves 4.5x (active-set)
  and 6.2x (Clarabel) faster with eight outer workers.
- Test families and scripts: `examples/stress_mpcc.py` (complementarity
  constraints), `examples/stress_linear.py`,
  `scripts/compare_corrections.py`, `scripts/compare_variants.py`.

### Changed

- 0.15 performance work (#11): cheaper validation, matrix products and
  BFGS; exact-bit Cholesky factor and inverse caches in the active-set
  backend; a checked closed-form path for unconstrained and diagonal-box
  QPs. On the delivery host the active-set path is
  1.12-1.50x faster on the four fixtures (geometric mean 1.247x);
  Clarabel wall times are mixed.
- The closed-form QP path changes active-set results on unconstrained
  problems in the last bits. On 1,036 CUTEst problems (review) 55 runs
  change numerically, 50 of them unconstrained, and the active-set backend
  solves 801 instead of 799. Clarabel and elastic mode give identical runs
  apart from runs near the time limit.

### Fixed

- Convergence is no longer claimed on residuals at their rounding level
  (#7). Stationarity and complementarity must clear their tolerances by
  `(k_j + 2) eps (|grad f_j| + sum_i |J_ij lambda_i| + |z_j|)` per entry
  and `|mu| eps (|v| + |b|)` per product. With Clarabel, QP multipliers of
  5e9 had produced a false convergence claim on the complementarity family
  (seed 1, case 80). The 9,000 stress runs of both backends are
  bit-identical with and without the guard.

## [0.14.0] - 2026-10-17

Ten performance iterations, reviewed (#5). The per-iteration versions 0.5.0
to 0.13.0 exist only as checkpoints of the delivery, not on `main`.

### Added

- `SqpOptions.hessian_block_size` (0, the default, keeps full BFGS):
  explicit block-diagonal damped BFGS. A failed block update uses one
  Hessian reset and switches once to full BFGS, counted in
  `RunReport.block_hessian_fallbacks`.
- `QpOutcome::NotPositiveDefinite` and `RunReport.qp_not_positive_definite`
  (also in Python). A rejected step QP resets B and re-solves the same QP
  at the same iterate without counting an iteration; repeated rejections
  end `numerical_failure` once the reset budget is spent.
- Native `performance` example and `scripts/compare_performance.py`, which
  pairs any number of labeled builds (it replaces
  `compare_performance_final.py`).

### Changed

- Breaking (custom `QpBackend` implementations): the driver no longer
  factorizes B. A backend that needs a positive-definite Hessian must check
  every model exactly and report `QpOutcome::NotPositiveDefinite`. The
  Clarabel adapter's exact check is the single gate and keeps its cached
  workspace and conversion data on a rejection. In review this replaced
  the delivered rank-updated BFGS factor, which let an indefinite B abort
  solves above 16 variables.
- Allocation and work reductions with full-mode results bit-identical to
  0.4.0 (6,000 stress runs, 26 comparison runs, orbit objectives): one
  BFGS triangle computed and mirrored in a reused workspace; one step-QP
  template per solve; reusable KKT-check buffers; recycled derivative
  output buffers, refilled with NaN before each callback; restoration QPs
  that keep exact sparsity; strictly diagonally dominant Hessians
  certified without Cholesky; cached conic conversion data and scatter
  maps.
- Orbit, 164 variables, workspace reuse (medians, baseline -> reviewed
  release): BFGS phase 95.6 -> 33.8 ms, BFGS plus conversion 163.9 ->
  105.2 ms. The Clarabel engine still dominates the wall time.

## [0.4.0] - 2026-10-16

Iteration 3: QP workspace reuse, reviewed (#4).

### Added

- Opt-in Clarabel workspace reuse: `SqpOptions.qp_reuse_workspace` (Rust
  and Python) and `QpOptions.reuse_workspace`, default false. A workspace
  is reused when dimensions, P/A CSC structure, cone kinds and the signed
  dual mapping match; all values are replaced and the iterates still start
  cold. A reused solution that is not independently accepted gets one cold
  retry, and a reused infeasibility finding needs cold confirmation before
  restoration. Clarabel's setup and post-processing timers are reset per
  call, so earlier calls do not count towards its time limit.
- QP phase telemetry: `qp_setups`, `qp_updates`, `qp_cold_retries`,
  `qp_retry_reason` and conversion, setup, update, solve and check
  seconds in the run report.
- `examples/compare_workspace.py`. Delivery host: median orbit wall time
  -9.9% (84 variables) and -15.6% (164) with reuse, same outer iterations,
  acceptance unchanged (13 of 13 cases, 1,186 of 1,200 stress problems).

### Changed

- QP conversion builds Clarabel's CSC data directly in O(nnz) instead of
  sorting triplets (orbit, 164 variables: about 0.145 -> 0.075 s per run);
  the cache key is compared in place; the BFGS update is computed
  entry-wise without five n x n temporaries. Results are bit-identical.

## [0.3.0] - 2026-10-16

Iteration 2: bounded feasibility restoration, reviewed (#3).

### Added

- Bounded feasibility restoration: after an infeasible step QP,
  regularized Gauss-Newton steps on the squared row violation within the
  variable bounds. An accepted point must decrease the violation and be
  completely evaluated; curvature and penalties are then reset.
  Restoration-QP multipliers never become NLP multipliers, and only the
  unchanged NLP KKT test declares convergence.
- Options `max_restoration_steps` (10; 0 gives the previous behavior),
  `restoration_step_limit` (1.0) and `restoration_regularization` (1e-4),
  in Rust and Python.
- Statuses `restoration_failed` and `restoration_limit`; report fields
  `restoration_attempts`, `restoration_steps`, `restoration_seconds`,
  `restoration_history` (`RestorationRecord`) and
  `bound_step_corrections`. Restoration stops as `restoration_failed` when
  a step not cut by the direction bound predicts less than `1e-8 * psi`.
- HS071 from (1, 1, 1, 1) converges (one restoration and seven SQP steps)
  instead of ending `linearization_infeasible`; the SLSQP comparison
  accepts 13 of 13 sqp-craft results (12 before).
- `examples/restore_hs071.py` and `examples/stress_random.py` (random
  problems feasible by construction, run in CI for seeds 1-4): 1,186 of
  1,200 converge, with no false convergence claim.

### Fixed

- A QP step overshooting a nearly active bound (room 3e-14, overshoot
  1e-11) cut the step length to about zero and stalled the solve. Step
  components that exceed their step bounds by at most the feasibility
  tolerance are moved onto the bound, and trial components that rounding
  of `x + alpha p` puts a few ulps beyond a bound are set to it.
- A step whose predicted merit decrease is non-positive only within the
  QP's accuracy ended `stalled` just short of convergence. It is now taken
  once at full length if the merit rises by no more than that accuracy,
  at most three times in a row.

## [0.2.0] - 2026-10-16

Iteration 1: native SQP pilot, reviewed (#2).

### Added

- Native SQP driver, crate `sqp-craft-solver` (`crates/sqp`), selected
  with `method="sqp"` (the default `"dummy"` still only evaluates the
  start): step QPs in shifted coordinates, dense damped BFGS with
  positive-definiteness checks and bounded resets, an l1 merit with
  Powell's per-row penalties `mu_i <- max(t_i, (mu_i + t_i)/2)`,
  `t_i = 1.1|lambda_i|`, Armijo backtracking with retries at undefined
  points and a first-order convergence test. Research pilot with at most
  256 variables (`dense_size_limit`).
- Clarabel 0.11.1 QP adapter, crate `sqp-craft-clarabel`: two-sided rows
  as cones, signed multipliers, and one scale-aware acceptance rule,
  `QpModel::accepts`, shared by adapter and driver.
- Derivative checker `check_derivatives`: coordinate finite differences
  inside the box with several step sizes, every output row compared;
  fixed variables are listed in `fixed` and do not affect `passed`, and an
  undefined perturbed point does not abort the check.
- Iteration, callback and time budgets; `RunReport`; optional QP trace and
  iteration history (`record_history`); typed Python results (`Status`
  literal, `DerivativeIssue`, `IterationRecord`); descriptive status
  messages. A non-positive predicted decrease within the QP's accuracy
  ends `stalled`.
- Numerical hardening: checked KKT intermediates (an overflow is an error,
  never a zero residual), `Bounds::checked_violation` and finite QP
  warm-start validation.
- Fixtures: orbit-transfer collocation with 84 and 164 variables (Rust
  test and Python example), HS071 and a bound-active quartic.
  `examples/compare_baseline.py` compares with SciPy SLSQP under one
  independent evaluator: sqp-craft is accepted on 12 of 13 cases (HS071
  from (1, 1, 1, 1) ends `linearization_infeasible`), SLSQP on 12.
- CI runs debug tests, the orbit example, the SLSQP comparison and a build
  without the native solver.

### Changed

- Python: native phases run detached from the interpreter, and the
  exception of a callback the solver recovered from (an undefined point)
  is dropped; only the most recent callback's exception is re-raised, and
  only for the matching solver error.

## [0.1.0] - 2026-10-15

Interface draft (#1). No optimizer: `DummySolver` evaluates the start point
once and returns status `"stub"`.

### Added

- Core crate `sqp-craft` without dependencies: validated, immutable
  `ProblemSpec` with field- and entry-specific errors; `NlpProblem` with
  two grouped callbacks, `functions(x)` for f and c and `derivatives(x)`
  for the gradient and the Jacobian values, written into solver buffers.
  Solvers never evaluate outside the variable bounds, and derivatives only
  at the point of the preceding `functions` call.
- Multiplier sign convention `grad f = J^T lambda + z`, independent KKT
  residuals (`kkt` module) and the `UndefinedPoint` error kind.
- QP seam with multipliers, `QpOutcome`, active-set warm start and an
  independent residual check; `DummyQp` never fabricates a step.
- Python package `sqp_craft` (PyO3 0.30.1, abi3 for CPython 3.10+):
  keyword-only, picklable `ProblemSpec`, NumPy-friendly input and output,
  `EvaluationError` and `UndefinedPointError` with the original exceptions
  re-raised; an observer returns `True` to stop.
- Docs (API, design notes with the review record, target card), a check
  script and GitHub Actions CI: fmt, Clippy, tests, MSRV 1.85, and Python
  3.10 and 3.14 on Linux, macOS and Windows.

Repository: https://github.com/dietmarwo/SQP-craft
