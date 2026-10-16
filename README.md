# SQP-craft

A native Rust SQP iteration loop with Python bindings. **Iteration 1 / v0.2.0**
implements damped dense BFGS, an L1 merit line search, grouped callbacks,
independent KKT checks and Clarabel QP subproblems.

This is a bounded analytic research pilot: **at most 256 variables**, one
thread per solve, positive-definite QPs and cold QP starts. Sparse input is
supported; the BFGS matrix is dense. There is no restoration or automatic
scaling, and a nonlinear feasible problem can have an infeasible linearization.

## Use from Python

```python
from sqp_craft import ProblemSpec, SqpOptions, check_derivatives, solve

class Toy:
    def functions(self, x):
        return (x[0]-1)**2 + (x[1]-2)**2, [x[0]+x[1]]
    def derivatives(self, x):
        return [2*(x[0]-1), 2*(x[1]-2)], [1.0, 1.0]

spec = ProblemSpec(lower=[0, 0], upper=[4, 4],
                   constraint_lower=[3], constraint_upper=[3],
                   jacobian_indptr=[0, 2], jacobian_indices=[0, 1])
assert check_derivatives(spec, Toy(), [0.5, 0.5]).passed
result = solve(spec, Toy(), [0.5, 0.5], method="sqp",
               method_options=SqpOptions(max_iterations=100))
assert result.status == "converged"
print(result.candidate, result.stationarity, result.report)
```

Select `method="sqp"` explicitly. The compatibility default `"dummy"` evaluates
only the start and returns `"stub"`. Convergence requires all four first-order
residual tests; it does not establish a global minimum.

## Layout

| Path | Role |
| --- | --- |
| `crates/core` | Dependency-free contracts, KKT checks, derivative checker and dummy |
| `crates/clarabel` | Clarabel 0.11.1 QP adapter with signed-dual mapping and cancellation |
| `crates/sqp` | Native SQP driver, damped BFGS and acceptance tests |
| `crates/python`, `python/sqp_craft` | Owned PyO3 callbacks, typed facade and stub |
| `examples/solve_sqp.py`, `crates/sqp/examples/solve.rs` | Real Python and Rust solves |
| `examples/compare_baseline.py` | Independent analytic comparison against SciPy SLSQP |
| [docs/api.md](docs/api.md) | Callback, QP, options, result and exception contracts |
| [docs/iteration-1-results.md](docs/iteration-1-results.md) | Implemented plan, validation and pilot limitations |

## Build and test

Requires Rust **1.85+**, Python **3.10+** and a C linker. The selected QP backend
is Rust code and requires no external optimizer, BLAS, CMake or libclang.
Cargo.lock pins the dependency graph.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'maturin>=1.9.4,<2' pytest numpy patchelf
bash scripts/check.sh
PYO3_BUILD_EXTENSION_MODULE=1 python -m maturin build --release --locked
# Optional independent analytic comparison
python -m pip install 'scipy>=1.15,<2'
python examples/compare_baseline.py > baseline.json
```

`scripts/check.sh` runs formatting, Clippy, release tests, Rust examples,
compiled Python tests and Python examples. `cargo test -p sqp-craft --locked`
tests the contracts without the solver. For a dummy-only extension add
`--no-default-features` to maturin. Missing native support raises an explicit error.

CI defines Linux/macOS/Windows Python jobs. This archive was validated on Linux,
not the complete CI matrix. No package is published (`publish = false`).

## License

Project code is MIT; see [LICENSE](LICENSE) and [docs/dependencies.md](docs/dependencies.md).
