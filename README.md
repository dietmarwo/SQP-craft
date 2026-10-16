# SQP-craft

A sequential quadratic programming (SQP) solver in Rust, with Python bindings,
for smooth constrained problems such as trajectory transcriptions.

**Version 0.3.0: a research pilot.** Dense damped BFGS, an L1 merit line search
with per-constraint penalties, Clarabel QP subproblems, grouped model callbacks
and independent KKT checks. Limits: at most 256 variables, one thread per
solve, bounded Gauss–Newton feasibility restoration, no second-order correction,
no automatic scaling. Restoration can fail at infeasible stationary points; it
never certifies nonlinear infeasibility. Set `max_restoration_steps=0` for the
previous infeasible-linearization stop behavior.
Convergence means the first-order KKT residuals are within tolerance at the
returned point, not a global optimum.

Iteration 2 builds on reviewed branch `iteration-1` at `e627721`, rather than
`main`'s older interface draft. It recovers the previously failing HS071 start
`[1,1,1,1]`, with unchanged independent acceptance tolerances. See the
[plan](docs/iteration-2-plan.md) and [results](docs/iteration-2-results.md).

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

## Layout

| Path | Role |
| --- | --- |
| `crates/core` (`sqp-craft`) | Dependency-free contracts: problem spec, callbacks, solver traits, results, KKT residuals, QP seam, derivative checker, dummy method |
| `crates/sqp` (`sqp-craft-solver`) | The native SQP driver and damped BFGS |
| `crates/clarabel` (`sqp-craft-clarabel`) | Clarabel 0.11.1 QP adapter |
| `crates/python`, `python/sqp_craft` | PyO3 extension and typed Python API |
| [docs/api.md](docs/api.md) | The contract: conventions, algorithm, options, statuses, errors |
| `examples/restore_hs071.py` | Compare disabled restoration with the recovered HS071 solve |
| [docs/iteration-2-results.md](docs/iteration-2-results.md) | Current implementation, measured acceptance and limits |
| [docs/iteration-1-results.md](docs/iteration-1-results.md) | What iteration 1 delivers, review findings, SLSQP comparison, limits |
| [docs/validation.md](docs/validation.md) | Checks run for this version |

## Build and test

Requirements: Rust 1.85 or newer, CPython 3.10 or newer, a C linker. The QP
backend is Rust code; no external optimizer, BLAS or CMake is needed.

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

## License

MIT, see [LICENSE](LICENSE) and [docs/dependencies.md](docs/dependencies.md).
