# Validation — iteration 2

Version 0.3.0, 16 October 2026 UTC. Verified base:
`e627721e73aad429ae6e9df7615c37f892ce6d7e` on `origin/iteration-1`.
Earlier evidence is retained in [iteration-1-validation.md](iteration-1-validation.md).

Local environment: Linux x86_64, Rust 1.99.1, CPython 3.12.14,
PyO3 0.30.1, maturin 1.9.4, pytest 9.1.1, NumPy 2.5.4 and SciPy 1.18.1.
Rust 1.85.0 was checked separately as the declared minimum.

| Completed check | Result |
| --- | --- |
| Original reviewed baseline | 57 Rust tests + 1 doctest; 67 Python tests pass |
| `cargo fmt --all --check` | Pass |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Pass |
| `cargo test --workspace --locked` (debug, overflow checks enabled) | **65 tests + 1 doctest pass** |
| `cargo +1.85.0 check --workspace --all-targets --locked` | Pass |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked` | Pass |
| Native release wheel build | Pass, `cp310-abi3-manylinux_2_34_x86_64` |
| Compiled Python suite in separate installed-wheel environment | **75 pass** |
| `mypy --strict python/sqp_craft` | Pass |
| `examples/restore_hs071.py` | Disabled: old failure; enabled: converges in 8 steps, including 1 restoration step |
| Same 13-case independent comparison | Native 13/13 accepted; SLSQP 12/13 accepted |
| Dummy-only wheel with `--no-default-features` | Builds; **42 contract tests pass**; native solve raises `NotImplementedError` |
| `git diff --check` | Pass |

The wheel-test interpreter imported sqp-craft from its `site-packages`, not the
source checkout or an editable install. Default and dummy-only wheels were
built into different directories. The source archive includes neither wheels
nor build directories. Its Cargo.lock pins external dependencies; only workspace
package versions changed from the reviewed dependency graph.

Reproduction from an activated virtual environment:

```bash
python -m pip install 'maturin>=1.9.4,<2' pytest numpy scipy mypy
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo +1.85.0 check --workspace --all-targets --locked
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
PYO3_BUILD_EXTENSION_MODULE=1 python -m maturin build --release --locked --out dist
# Install the generated wheel in a separate virtual environment:
python -m pip install dist/*.whl pytest numpy scipy
python -m pytest -q tests
python examples/restore_hs071.py
python examples/compare_baseline.py > comparison.json
```

The explicit PyO3 environment setting supports the locally used maturin 1.9.4;
the build emitted a legacy extension-feature warning, but the installed wheel
was exercised successfully. Python >=3.10 is the ABI target, not a statement
that every supported interpreter was run locally. The existing CI matrix covers
Python 3.10/3.14 on Linux/macOS/Windows; those remote jobs were not run for these
local changes. The updated CI includes the restoration example and new tests.

The numerical regression tests retain the previous acceptance thresholds. New
coverage includes an independent HS071 KKT check, permanently inconsistent rows,
fixed coordinates, disabled restoration and distinct limit statuses, preservation
of the previous point on partial trials, observer/cancellation stops, recovered
undefined functions/derivatives, original fatal exception identity, nonfinite
output, invalid controls and Gauss–Newton arithmetic overflow.

No application dataset, 16-core scaling, free-threaded Python, global feasibility
guarantee or runtime speedup was validated. See the complete
[iteration-2 results](iteration-2-results.md) for interpretation and limits.
