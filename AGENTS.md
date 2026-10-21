# AGENTS.md

SQP-craft is a sequential quadratic programming solver in Rust with Python
bindings, for smooth constrained NLPs with at most 256 variables (dense
damped BFGS, Clarabel or dense active-set step QPs, independent KKT checks).
It also builds `sqp-job-worker`, a worker for the shared job API of
[optimization-jobs](https://github.com/dietmarwo/optimization-jobs).

Read [ai-context.md](ai-context.md) first: what the solver is for, how to
formulate a problem, options, statuses, the job profiles and how to report
results. To choose between this and the sibling solvers (fcmaes-rust,
fast-cma-es, LP-IP-craft), start at
https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md.

## Layout

- `crates/core` (`sqp-craft`), `crates/sqp` (`sqp-craft-solver`),
  `crates/clarabel`, `crates/activeset`, `crates/parallel`: the library crates
- `crates/python`, `python/sqp_craft`: the Python package; `tests/`: its tests
- `crates/sqp-job-worker`: the job worker library (`Registry`, `Adapter`),
  the stock binary and `examples/custom_worker.rs`
- `tutorials/job-api`: runnable job tutorial; `docs/`: API, benchmarks, guides

## Build and test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
python -m maturin develop --release --locked && python -m pytest -q   # in a venv
cargo build --release --locked -p optimization-runner -p sqp-job-worker
cargo build --release --locked -p sqp-job-worker --example custom_worker
python3 tutorials/job-api/run.py --root "$(mktemp -d)"
```

`scripts/check.sh` runs the main Rust and Python checks; `ai-context.md`
lists the shared job harness commands that CI runs.

## Conventions

- MSRV: Rust 1.85 for the library crates, 1.88 for the job layer
  (`sqp-job-worker` and the optimization-jobs crates). Edition 2024.
- Always build with `--locked`. The shared job crates come from
  optimization-jobs at the git revision the root `Cargo.toml` pins; change
  them there first.
- Never weaken a tolerance or an independent check to make a test pass. A
  `converged` status is a first-order claim, not an optimum.
- Jobs never carry callbacks: new models are registered in a worker.
- Add a `CHANGELOG.md` entry under `[Unreleased]` for user-visible changes.
