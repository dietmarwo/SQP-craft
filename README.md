<p align="center">
  <img src="https://raw.githubusercontent.com/dietmarwo/SQP-craft/main/docs/images/sqp-craft.png" alt="SQP-craft: smooth constrained optimization in Rust">
</p>

# SQP-craft

A sequential quadratic programming (SQP) solver in Rust, with Python bindings,
for smooth constrained problems such as trajectory transcriptions. The current
dense-BFGS solver accepts at most 256 variables and needs derivatives.

Seven [application tutorials](tutorials/README.md) (SQP1–SQP6, X2) cover
constrained calibration, multistart and scaling, a hanging chain, collocation
from the brachistochrone to a low-thrust transfer, an Earth–Mars Sims–Flanagan
leg, reaction kinetics and AC power dispatch, each with an independent
reference.

**Choosing an optimization tool?** Start with the
[shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md) or the
[optimization ecosystem overview](https://dietmarwo.github.io/optimization-jobs/).
The guide compares this solver with Python/Rust black-box optimization
and the other structured solver, then explains library, shell and MCP use.

![Pure Rust solver](https://img.shields.io/badge/solver-100%25%20Rust-brightgreen)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)
](https://github.com/rust-secure-code/safety-dance/)
![No C/C++ dependencies](https://img.shields.io/badge/C%2FC%2B%2B%20dependencies-none-brightgreen)
[![crates.io](https://img.shields.io/crates/v/sqp-craft-solver.svg?cacheSeconds=300)](https://crates.io/crates/sqp-craft-solver)
[![docs.rs](https://docs.rs/sqp-craft-solver/badge.svg)](https://docs.rs/sqp-craft-solver)
[![PyPI](https://img.shields.io/pypi/v/sqp-craft.svg?cacheSeconds=300)](https://pypi.org/project/sqp-craft/)
[![AI agent guide](https://img.shields.io/badge/guide-AI%20agents-blue)](ai-context.md)

> **Shared agent job API:** this repository builds `sqp-job-worker`, a native
> solver worker for the job API of
> [optimization-jobs](https://github.com/dietmarwo/optimization-jobs), at the
> revision that `Cargo.toml` pins. The same build makes the runner `optimization-run`
> and `optimization-mcp`, an MCP server through which agents submit, follow
> and cancel solves. See the [API guide](docs/job-api.md), the
> [runnable tutorial](tutorials/job-api/README.md), the
> [validation report](docs/job-api/VALIDATION.md) and the
> [review](docs/job-api/REVIEW.md). Existing solver APIs remain available.
> Your own NLPs run as jobs in a worker of your own built on the worker
> library ([application models](docs/job-api/models.md)).
> Sibling workers on the same node: [LP-IP-craft](https://github.com/dietmarwo/LP-IP-craft) for LP/MIP, [fcmaes-rust](https://github.com/dietmarwo/fcmaes-rust) for black-box, multimodal and multi-objective search.
> [experiment-craft](https://github.com/dietmarwo/experiment-craft) runs reproducible campaigns of these jobs (parameter sweeps, dependent steps, budgets and reports); its baseline rebuilds and tests `sqp-job-worker` at a pinned revision. It is in early development.

> **AI agents:** read [ai-context.md](ai-context.md) (formulation, options,
> results, multistart, the job API) and [AGENTS.md](AGENTS.md) (build, test,
> conventions).

**Version 0.16.0: eight-worker execution and thirty further experiments**
(iterations 31–60). Native `Options(threads=8)` offers a maximum worker budget;
large active-set QP outputs and opt-in native propagation models can use it.
Small pilot kernels and full BFGS keep their measured scalar path. Rust
`batch::solve_many` / `try_solve_many` run ordered independent starts with
worker-local models/backends, shared cancellation and an eight-CPU total budget.
Python callbacks/observers remain on their calling thread; the Clarabel engine
remains serial. Default threads=1 and the 256-variable NLP limit are unchanged.
See [the parallel plan](docs/performance-60/PLAN.md),
[results](docs/performance-60/RESULTS.md),
[reproduction/API guide](docs/performance-60/REPRODUCE.md) and
[the review before merging](docs/performance-60/REVIEW.md).

**Previous version 0.15.0: thirty further performance iterations** build on main
`4d776bb`. They reduce validation and symmetric-product work, streamline BFGS,
reuse exact-matching active-set factors, and independently check direct
unconstrained/diagonal-box QP proposals before accepting them. Solver defaults,
callbacks, numerical tolerances and the 256-variable pilot scope remain.
See the [thirty plans and checkpoints](docs/performance-30/PLAN.md),
[measured results](docs/performance-30/RESULTS.md) and
[reproduction commands](docs/performance-30/REPRODUCE.md). Improvements are
workload-dependent; raw slower/noisy observations are retained.

**Previous version 0.14.0: ten tested performance iterations** based on reviewed
`main` at `73d8d0c`. This is still a research pilot with at most 256 variables
and defaults to one numerical thread per solve. It uses damped BFGS, QPs solved by
Clarabel or by a dense active-set method, an L1 merit line search, bounded
feasibility restoration and independent first-order KKT checks. There is no
second-order/global optimality certificate or nonlinear infeasibility
certificate.

The performance work reuses numerical/checking/callback buffers, caches QP and
conic conversion structure, reduces symmetric update work and preserves exact
sparsity in restoration. QP adapters check positive definiteness of every new BFGS matrix. The
active-set adapter may reuse a successful factor only for bit-identical
Hessian structure and values. A failed check
resets it, like a failed update. Full BFGS remains the default. An explicit
`SqpOptions(hessian_block_size=...)` requests contiguous curvature blocks;
failed block updates switch once to full BFGS within the existing reset budget.
`report.block_hessian_fallbacks` records that switch. Block mode is an
approximation, and can improve or worsen solve time depending on the model
and variable order.

**A dense active-set QP backend** (Goldfarb–Idnani dual method, warm-started
from the previous working set) is available with
`SqpOptions(qp_solver="active-set")`. On the benchmark fixtures it takes the
same SQP steps as Clarabel and makes the whole solve 5.4 to 6.4 times faster
(the 164-variable orbit transfer: 2.99 s → 0.55 s). It gave the same status
as Clarabel in all 4,500 paired stress runs. Clarabel remains the default.
See [active-set-qp.md](docs/active-set-qp.md), and
[gaps.md](docs/gaps.md) for the ranked weaknesses that motivated it. With
that backend, `SqpOptions(qp_solver="active-set", elastic_mode=True)` solves
every step QP as an ℓ1-elastic QP instead of entering feasibility
restoration ([elastic-mode.md](docs/elastic-mode.md)); it is an option, not
the default, because it does not reduce the local failures.
`SqpOptions(second_order_correction=True)` tries one corrected step before
backtracking when a curved constraint makes the merit reject the full step
([second-order-correction.md](docs/second-order-correction.md)): 21% fewer
callbacks on the Maratos example, 1% on the stress family.

**Four opt-in mechanisms from SNOPT** ([snopt-lessons.md](docs/snopt-lessons.md)):
- **Curvature repair:** `hessian_update="augmented"` repairs small BFGS
  curvature with SNOPT's augmented-Lagrangian term instead of damping it.
- **Smooth merit:** `merit="augmented-lagrangian"` is SNOPT's merit function,
  and `violation_limit` adds its limit on trial-point violation.
- **Linear rows:** `ProblemSpec(..., linear_rows=[...])` keeps declared rows
  satisfied, as SNOPT does with linear constraints.
- **Scaling:** `scaling="linear"` or `"jacobian"` scales the QP and the BFGS
  matrix by powers of two.

Each helps on its own mechanism and costs elsewhere, so all stay off. Jacobian
scaling, for example, solves all 40 badly scaled orbit transfers instead of 20.
On well-scaled problems it takes 2–24% more iterations.

**A CUTEst benchmark** ([benchmark-results.md](docs/benchmark-results.md))
runs 1,036 problems from S2MPJ against SciPy SLSQP and Uno, with one
independent first-order check for every solver:
- **sqp-craft:** the default (Clarabel) solves 754 problems, the active-set
  backend 799, and 813 with elastic mode. It made no false claim.
- **The comparators:** Uno's filterSQP preset solves 760 with L-BFGS and 845
  with exact Hessians; SLSQP solves 631.

The results recommend the active-set backend with elastic mode as the
default; the defaults stay unchanged until that is decided. They were measured
with 0.14; the [0.16 review](docs/performance-60/REVIEW.md) reran the harness:
Clarabel and elastic mode give the same results, and the active-set backend
solves 801.

See [plans and results](docs/performance/README.md) for all ten checkpoints,
raw per-iteration measurements, final paired timings, numerical failure fixes
and independent validation. Numerical tolerances and model callbacks were
not relaxed or rewritten. The actual user trajectory application is still
unavailable; the orbit fixture and generated problems are development tests.

The [prompt-driven curve-fitting tutorial](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic/curve-fit) runs the existing constrained decay-fit worker and checks its half-sum-of-squares objective, anchor equality and first-order evidence.

The [prompt-driven SQP tutorial](https://github.com/dietmarwo/optimization-jobs/tree/main/tutorials/agentic/sqp) adds a checked agent workflow over the existing `sqp-job-worker`.

## Use from Python

```python
from sqp_craft import ProblemSpec, SqpOptions, check_derivatives, solve

class Toy:
    """minimize (x0-1)^2 + (x1-2)^2  subject to  x0 + x1 = 3,  0 <= x <= 4"""
    def functions(self, x):            # -> (objective, constraint values)
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, [x[0] + x[1]]
    def derivatives(self, x):          # -> (gradient, Jacobian values in CSR order)
        return [2 * (x[0] - 1), 2 * (x[1] - 2)], [1.0, 1.0]

spec = ProblemSpec(lower=[0, 0], upper=[4, 4],
                   constraint_lower=[3], constraint_upper=[3],
                   jacobian_indptr=[0, 2], jacobian_indices=[0, 1])
assert check_derivatives(spec, Toy(), [0.5, 0.5]).passed
result = solve(spec, Toy(), [0.5, 0.5], method="sqp",
               method_options=SqpOptions(record_history=True))
assert result.status == "converged"
print(result.candidate, result.multipliers, result.stationarity)
```

Pass `method="sqp"` explicitly: the default `"dummy"` only evaluates the start
point and returns `"stub"`. Callbacks may return lists, tuples or 1-D NumPy
arrays. `examples/orbit_transfer.py` solves a 164-variable low-thrust
collocation problem; `examples/compare_baseline.py` compares the solver with
SciPy's SLSQP under one independent evaluator, and `examples/stress_random.py`
does the same on random feasible problems from random starts.

To use the dense active-set QP backend, set
`SqpOptions(qp_solver="active-set")`; `qp_warm_start=True` (the default)
starts each QP from the previous working set.

To reuse compatible Clarabel allocations and symbolic structure, set
`SqpOptions(qp_reuse_workspace=True)`. The default is `False`; every QP still
starts its primal/dual iterates cold. `result.report` exposes `qp_setups`,
`qp_updates`, `qp_cold_retries` and separate conversion, setup, update, solve
and check times. See the [API](docs/api.md#qp-workspace-reuse).

## Layout

| Path | Role |
| --- | --- |
| `crates/core` (`sqp-craft`) | Dependency-free contracts: problem spec, callbacks, solver traits, results, KKT residuals, QP seam, derivative checker, dummy method |
| `crates/sqp` (`sqp-craft-solver`) | The native SQP driver and damped BFGS |
| `crates/clarabel` (`sqp-craft-clarabel`) | Clarabel 0.11.1 QP adapter (default backend) |
| `crates/activeset` (`sqp-craft-activeset`) | Dense dual active-set (Goldfarb–Idnani) QP backend; depends only on the core crate and `sqp-craft-parallel` |
| `crates/parallel` (`sqp-craft-parallel`) | Explicit bounded execution resources (Rayon pools) for the large QP kernels and native models |
| `crates/sqp-job-worker` | Job worker library (`Registry` of application models, `Adapter`) and the stock binary for the profiles `sqp.solve/1` and `sqp.evaluate/1` of the shared job API; `examples/custom_worker.rs` is an application worker ([models guide](docs/job-api/models.md)) |
| [docs/gaps.md](docs/gaps.md) | Weaknesses compared with mature SQP codes, and the order of work |
| `bench/` | CUTEst benchmark harness: S2MPJ access, adapter, solvers, independent check, isolated workers, report |
| [docs/benchmark-results.md](docs/benchmark-results.md) | CUTEst results: the open defaults, the comparators, the failure classes |
| [docs/active-set-qp.md](docs/active-set-qp.md) | The active-set backend: method, measurements, validation |
| [docs/elastic-mode.md](docs/elastic-mode.md) | Elastic mode: the elastic QP, the driver rules, measurements |
| `examples/stress_mpcc.py` | Random problems with complementarity constraints (degenerate QPs) |
| [docs/second-order-correction.md](docs/second-order-correction.md) | The second-order correction: rule, measurements |
| `scripts/compare_corrections.py` | Paired runs with and without the second-order correction |
| `crates/python`, `python/sqp_craft` | PyO3 extension and typed Python API |
| [docs/api.md](docs/api.md) | The contract: conventions, algorithm, options, statuses, errors |
| `examples/restore_hs071.py` | Compare disabled restoration with the recovered HS071 solve |
| [docs/performance/README.md](docs/performance/README.md) | Ten performance plans, implementation checkpoints, raw measurements and release validation |
| [docs/iteration-3-results.md](docs/iteration-3-results.md) | Reviewed baseline and previous performance profile |
| `examples/compare_workspace.py` | Repeated cold/reuse comparison with independent acceptance checks |
| [docs/iteration-2-results.md](docs/iteration-2-results.md) | Feasibility restoration and subsequent review fixes |
| [docs/iteration-1-results.md](docs/iteration-1-results.md) | What iteration 1 delivers, review findings, SLSQP comparison, limits |
| [docs/validation.md](docs/validation.md) | Checks run for this version |
| [ai-context.md](ai-context.md) | Operational context for AI agents: formulation, options, results, multistart, job API |
| [AGENTS.md](AGENTS.md) | Build, test and conventions for coding agents |
| `examples/multistart_processes.py` | Many starts of a Python model in separate processes, with a KKT recheck |

## Install from source

From 0.17.0, releases go to PyPI and crates.io:
`python -m pip install sqp-craft` installs a wheel (CPython 3.10 or newer on
Linux x86_64, Windows x64 and macOS x86_64/arm64) and needs no Rust; the Rust
crates are below. The job worker is source-only
([RELEASING.md](RELEASING.md) describes the releases). To build from source,
for example for changes after the latest release, you need git, Rust
1.85 or newer (`rustup` from https://rustup.rs; 1.88 for the job worker and the
runner), CPython 3.10 or newer, and a C linker (the platform's default one).
The QP backends are Rust code; no external optimizer, BLAS or CMake is needed.

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install .                        # Python package sqp_craft; pip fetches maturin
cargo test -p sqp-craft -p sqp-craft-solver --locked
```

Our own dependency: the job worker `sqp-job-worker` depends on the shared job
API of [optimization-jobs](https://github.com/dietmarwo/optimization-jobs)
through git at the revision pinned in `Cargo.toml`. Cargo fetches it on the
first build (network access to GitHub needed; `cargo fetch --locked`
prepares an offline build), so no second checkout is needed:

```sh
cargo build --release --locked -p optimization-runner -p sqp-job-worker
```

The library crates and the Python package depend on crates.io crates only.
A Rust project depends on the core (`ProblemSpec`, `NlpProblem`, results),
the solver and at least one QP backend:

```sh
cargo add sqp-craft sqp-craft-solver sqp-craft-clarabel   # add sqp-craft-activeset for ActiveSetQp
```

For changes after the latest release, use git dependencies instead:

```toml
[dependencies]
sqp-craft = { git = "https://github.com/dietmarwo/SQP-craft" }
sqp-craft-solver = { git = "https://github.com/dietmarwo/SQP-craft" }
sqp-craft-clarabel = { git = "https://github.com/dietmarwo/SQP-craft" }    # ClarabelQp, the default
sqp-craft-activeset = { git = "https://github.com/dietmarwo/SQP-craft" }   # ActiveSetQp, optional
```

## Build and test

Requirements: Rust 1.85 or newer, CPython 3.10 or newer, a C linker. The QP
backends are Rust code; no external optimizer, BLAS or CMake is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install 'maturin>=1.9.4,<2' pytest numpy
bash scripts/check.sh
```

`scripts/check.sh` runs rustfmt, Clippy with warnings denied, the Rust tests,
both Rust examples, builds the extension, and runs the Python tests and
examples. `python -m pip install .` builds and installs the package;
`python -m maturin build --release` builds an abi3 wheel, and
`--no-default-features` builds one without the native solver. The core crate
works without Python: `cargo test -p sqp-craft`. For the SLSQP comparison:
`python -m pip install scipy && python examples/compare_baseline.py --table`
(and `python examples/stress_random.py --table` for the stress test).
Use `python examples/stress_random.py --reuse-workspace --table` to exercise
reuse, `--qp-solver active-set` for the active-set backend, and `python examples/compare_workspace.py --repeats 5 > workspace-results.json`
for the alternating-order timing comparison. These commands require SciPy.
For the CUTEst benchmark: `python -m pip install ".[bench]" unopy`, then the
commands at the end of [benchmark-results.md](docs/benchmark-results.md)
(`python -m bench.s2mpj fetch` first).

## License

MIT, see [LICENSE](LICENSE) and [docs/dependencies.md](docs/dependencies.md).

## Performance experiments

```bash
cargo build --release -p sqp-craft-solver --example performance --locked
cargo run --release -p sqp-craft-solver --example performance --locked -- orbit 40 reuse 0
cargo run --release -p sqp-craft-solver --example performance --locked -- separable 128 reuse 2
cargo run --release -p sqp-craft-solver --example performance --locked -- orbit 40 reuse 0 active-set
python3 scripts/performance_iteration.py target/release/examples/performance measurements.json
python3 scripts/compare_performance.py pair.json base=PATH_A new=PATH_B --blocks new
python3 scripts/compare_performance.py backends.json clarabel=PATH active-set=PATH:active-set
```

The final comparison includes cold and reused QP workspaces, plus explicit
block/hybrid runs, with fixed callbacks and the same reference objectives and
original-unit KKT thresholds. Use `qp_reuse_workspace=True` to reproduce the
reuse measurements; the default remains `False`. Timings are local fixture
results and are not guaranteed to improve for every problem or iteration.
