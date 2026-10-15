# SQP-craft

Building blocks for a sparse sequential quadratic programming (SQP) solver in
Rust, with Python bindings.

> **Status: interface draft. No optimizer is implemented yet.**
> The only backend evaluates the start point once and returns it with
> `status == "stub"`. It exists so the problem, callback, result and
> cancellation contracts can be exercised end to end, from Rust and from
> Python, before a real SQP backend is integrated.

## What is here

| Component | Purpose |
| --- | --- |
| `crates/core` (`sqp-craft`) | Dependency-free Rust contracts: validated `ProblemSpec`, the `NlpProblem` callback trait, `SqpSolver`/`Observer` traits, typed results, KKT residuals, a provisional QP-subproblem seam, and `DummySolver` |
| `crates/python` | PyO3 extension module `sqp_craft._native` |
| `python/sqp_craft` | Typed Python API (`solve`, `ProblemSpec`, `Options`, result dataclasses) |
| `docs/api.md` | The contract: data conventions, callback guarantees, statuses, errors |
| `docs/design-notes.md` | Decisions, what is deferred, and the next steps |
| `docs/target-card.md` | Application inputs still to be supplied (left TBD on purpose) |

## Python example

```python
from sqp_craft import ProblemSpec, solve

class Toy:
    """minimize (x0-1)^2 + (x1-2)^2  subject to  x0 + x1 = 3,  0 <= x <= 4"""
    def functions(self, x):            # -> (objective, constraint values)
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, [x[0] + x[1]]
    def derivatives(self, x):          # -> (gradient, Jacobian values in CSR order)
        return [2 * (x[0] - 1), 2 * (x[1] - 2)], [1.0, 1.0]

spec = ProblemSpec(lower=[0, 0], upper=[4, 4],
                   constraint_lower=[3], constraint_upper=[3],
                   jacobian_indptr=[0, 2], jacobian_indices=[0, 1])
result = solve(spec, Toy(), [1.0, 2.0], observer=print)
assert result.status == "stub"          # never a convergence claim
```

Callbacks may return lists, tuples or 1-D NumPy arrays. The Rust equivalent is
in the crate documentation (`cargo doc --open -p sqp-craft`) and in
`crates/core/examples/toy.rs`.

## Build and test

Requirements: stable Rust (edition 2024, 1.85 or newer), a C linker, and
CPython 3.10 or newer. No external optimization libraries are needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install 'maturin>=1.9.4,<2' pytest numpy
bash scripts/check.sh
```

`scripts/check.sh` runs `cargo fmt --check`, `cargo clippy -D warnings`,
`cargo test`, builds the extension with `maturin develop`, then runs `pytest`
and both examples. `python -m pip install .` builds and installs the package;
`python -m maturin build --release` builds an abi3 wheel. The core crate is
usable without Python: `cargo test -p sqp-craft`.

The binding crate has no Rust test executable (it would need to link
libpython); it is tested through pytest against the compiled module.

## License

MIT, see [LICENSE](LICENSE).
