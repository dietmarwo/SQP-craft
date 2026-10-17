# SQP-craft

A sequential quadratic programming (SQP) solver in Rust, with Python bindings,
for smooth constrained problems such as trajectory transcriptions.

**Version 0.14.0: ten tested performance iterations** based on reviewed
`main` at `73d8d0c`. This is still a research pilot with at most 256 variables
and one numerical thread per solve. It uses damped BFGS, Clarabel QPs, an L1
merit line search, bounded feasibility restoration and independent first-order
KKT checks. There is no second-order/global optimality certificate or nonlinear
infeasibility certificate.

The performance work reuses numerical/checking/callback buffers, caches QP and
conic conversion structure, reduces symmetric update work and preserves exact
sparsity in restoration. Each BFGS matrix is checked for positive
definiteness exactly once per iteration, by the QP adapter. A failed check
resets it, like a failed update. Full BFGS remains the default. An explicit
`SqpOptions(hessian_block_size=...)` requests contiguous curvature blocks;
failed block updates switch once to full BFGS within the existing reset budget.
`report.block_hessian_fallbacks` records that switch. Block mode is an
approximation, and can improve or worsen solve time depending on the model
and variable order.

See [plans and results](docs/performance/README.md) for all ten checkpoints,
raw per-iteration measurements, final paired timings, numerical failure fixes
and independent validation. Numerical tolerances and model callbacks were
not relaxed or rewritten. The actual user trajectory application is still
unavailable; the orbit fixture and generated problems are development tests.

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

To reuse compatible QP allocations and symbolic structure, set
`SqpOptions(qp_reuse_workspace=True)`. The default is `False`; every QP still
starts its primal/dual iterates cold. `result.report` exposes `qp_setups`,
`qp_updates`, `qp_cold_retries` and separate conversion, setup, update, solve
and check times. See the [API](docs/api.md#qp-workspace-reuse).

## Layout

| Path | Role |
| --- | --- |
| `crates/core` (`sqp-craft`) | Dependency-free contracts: problem spec, callbacks, solver traits, results, KKT residuals, QP seam, derivative checker, dummy method |
| `crates/sqp` (`sqp-craft-solver`) | The native SQP driver and damped BFGS |
| `crates/clarabel` (`sqp-craft-clarabel`) | Clarabel 0.11.1 QP adapter |
| `crates/python`, `python/sqp_craft` | PyO3 extension and typed Python API |
| [docs/api.md](docs/api.md) | The contract: conventions, algorithm, options, statuses, errors |
| `examples/restore_hs071.py` | Compare disabled restoration with the recovered HS071 solve |
| [docs/performance/README.md](docs/performance/README.md) | Ten performance plans, implementation checkpoints, raw measurements and release validation |
| [docs/iteration-3-results.md](docs/iteration-3-results.md) | Reviewed baseline and previous performance profile |
| `examples/compare_workspace.py` | Repeated cold/reuse comparison with independent acceptance checks |
| [docs/iteration-2-results.md](docs/iteration-2-results.md) | Feasibility restoration and subsequent review fixes |
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
Use `python examples/stress_random.py --reuse-workspace --table` to exercise
reuse, and `python examples/compare_workspace.py --repeats 5 > workspace-results.json`
for the alternating-order timing comparison. These commands require SciPy.

## License

MIT, see [LICENSE](LICENSE) and [docs/dependencies.md](docs/dependencies.md).

## Performance experiments

```bash
cargo build --release -p sqp-craft-solver --example performance --locked
cargo run --release -p sqp-craft-solver --example performance --locked -- orbit 40 reuse 0
cargo run --release -p sqp-craft-solver --example performance --locked -- separable 128 reuse 2
python3 scripts/performance_iteration.py target/release/examples/performance measurements.json
python3 scripts/compare_performance.py pair.json base=PATH_A new=PATH_B --blocks new
```

The final comparison includes cold and reused QP workspaces, plus explicit
block/hybrid runs, with fixed callbacks and the same reference objectives and
original-unit KKT thresholds. Use `qp_reuse_workspace=True` to reproduce the
reuse measurements; the default remains `False`. Timings are local fixture
results and are not guaranteed to improve for every problem or iteration.
