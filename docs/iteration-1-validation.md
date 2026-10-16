# Validation

Version 0.2.0, 16 October 2026. Local runs on Linux x86_64 with Rust 1.97.0,
CPython 3.13, PyO3 0.30.1, maturin 1.15.0, pytest 9.1.1, NumPy 2.5.4 and
SciPy 1.18.1. The CI workflow repeats these checks on every pull request,
including Rust 1.85 (the declared minimum) and Python 3.10 / 3.14 on Linux,
macOS and Windows.

| Check | Result |
| --- | --- |
| `bash scripts/check.sh` from a clean build | Pass |
| `cargo fmt --all --check` | Pass |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Pass |
| `cargo test --workspace --locked` (debug, overflow checks on) | 57 tests + 1 doctest pass |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` | Pass |
| Rust examples `toy` and `solve` | Pass |
| Fresh virtual environment, `pip install ".[test]"`, `pytest` | 67 pass |
| README example, `examples/solve_sqp.py`, `examples/orbit_transfer.py 20` | Pass |
| Build with `--no-default-features`, `pytest tests/test_contract.py` | 42 pass; `method="sqp"` raises `NotImplementedError` |
| `mypy --strict` on `python/sqp_craft` | Pass |
| `examples/compare_baseline.py` | sqp-craft accepted on 12 of 13 cases, SLSQP on 12 of 13; both fail HS071 from (1, 1, 1, 1) |

## What the tests cover

- **Contracts** (core): problem spec validation, callback order and buffers,
  the dummy method, the multiplier sign convention on hand-derived cases,
  overflow in KKT arithmetic, QP warm-start validation.
- **QP adapter**: off-diagonal Hessian storage, signed lower/upper/fixed
  multipliers, infeasible rows, iteration limits, cancellation from inside the
  running Clarabel iteration, warm-start rejection, and scale-aware acceptance
  with a multiplier near −999 that still rejects a flipped sign.
- **Driver**: analytic optima with signed multipliers; an inconsistent
  linearization reported as such; recovered undefined trials; callback budgets
  that return the last complete point; observer stop and time limit; a quartic
  with an active bound and large early multipliers; HS071 from three starts;
  the iteration history; and the N = 20 orbit transfer, which must converge
  to r(tf) = 1.52206 in under 150 iterations with mostly full steps.
- **Derivative checker**: injected sign and pattern errors, fixed variables,
  `max_directions`, undefined perturbations, and a point 1e-13 inside a bound.
- **Python boundary**: the above through the compiled module, plus exception
  identity, a non-finite output after a recovered undefined trial, box
  assertions inside callbacks, observer events per accepted step, and solves in
  spawned worker processes.

Four of the new tests fail on the delivered iteration-1 code: the QP-acceptance
case (`qp_failure`), both orbit-transfer tests (iteration limit; derivative
check not passed because of the fixed initial state) and the Python derivative
check with an undefined perturbation (raised `UndefinedPointError`).

## Not validated

No application model or data (the target card is open), no 16-core or
multi-process scaling measurement, no free-threaded Python, no timing claims:
the comparison runs each case once.
