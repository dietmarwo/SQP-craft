# Validation — iteration 2

Version 0.3.0, 16 October 2026. Base: `main` at `2becc56` (the squash merge of
the reviewed iteration 1, same tree as `e627721`). Earlier evidence:
[iteration-1-validation.md](iteration-1-validation.md).

Local runs on Linux x86_64 with Rust 1.97.0, CPython 3.13, PyO3 0.30.1,
maturin 1.15.0, pytest 9.1.1, NumPy 2.5.4 and SciPy 1.18.1. The CI workflow
repeats these checks on every pull request, including Rust 1.85 (the declared
minimum) and Python 3.10 / 3.14 on Linux, macOS and Windows.

| Check | Result |
| --- | --- |
| Delivered iteration 2, `bash scripts/check.sh` | Pass: 65 Rust tests + 1 doctest, 75 Python tests |
| Reviewed, `bash scripts/check.sh` from a clean build | Pass |
| `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings` | Pass |
| `cargo test --workspace --locked` (debug, overflow checks on) | **70 tests + 1 doctest pass** |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked` | Pass |
| Fresh virtual environment, `pip install ".[test]"`, `pytest` (imported from `site-packages`) | **76 pass** |
| `mypy --strict python/sqp_craft` | Pass |
| `examples/restore_hs071.py` | Restoration off: `linearization_infeasible`; on: converged in 8 iterations (1 restoration step) |
| Build with `--no-default-features`, `pytest tests/test_contract.py` | 42 pass; `method="sqp"` raises `NotImplementedError` |
| `examples/compare_baseline.py` | sqp-craft 13 / 13 accepted, SLSQP 12 / 13; unchanged from the delivery |
| `examples/stress_random.py`, seeds 1–4, 300 problems each | 1186 / 1200 converged and accepted (delivered: 1184); no convergence claim rejected by the evaluator |

## New tests in the review

- `crates/sqp/tests/random_instances.rs`: the three stress instances the
  delivered code mishandled — a step overshooting a nearly active bound
  (`stalled` before), a step at the QP noise level (`stalled` at stationarity
  1.18e-6 before) and an infeasible stationary point of ψ
  (`restoration_limit` before). All three fail on the delivered code with
  exactly those statuses, and pass now with independent KKT checks and the
  SLSQP objectives.
- Unit tests for the trial-point rounding (a real case where `x + αp` lands
  one ulp above the bound) and for snapping only tolerance-level overshoots.
- The permanently inconsistent restoration fixture now requires
  `restoration_failed` after exactly two steps at the minimizer of ψ, in Rust
  and through Python, instead of accepting either restoration status.

## Not validated

No application model or data (the target card is open), no 16-core or
multi-process scaling measurement, no free-threaded Python, no timing claims.
The stress problems are one random family; they exercise inconsistent
linearizations and active bounds, not trajectory structure.
