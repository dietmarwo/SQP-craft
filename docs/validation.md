# Validation — iteration 3

Version 0.4.0 · 16 October 2026 UTC. Base: reviewed `main` at
`214f93b0ec0cf407f62fd01ef23df90c7a81fe5c`.
Historical evidence: [iteration 2](iteration-2-validation.md).
Current [results](iteration-3-results.md) and [plan](iteration-3-plan.md).

Local runs used Linux x86_64, Rust 1.99.1 (and Rust 1.85 for the minimum-version
check), CPython 3.12.14, PyO3 0.30.1, maturin 1.9.4, NumPy 2.5.4 and SciPy 1.18.1.
The native release wheel was installed in a fresh virtual environment; tests
imported `sqp_craft` from that environment's `site-packages`. The dummy-only
wheel was installed into a separate import directory. No prebuilt binary is
included in the source ZIP.

| Check | Result |
| --- | --- |
| Reviewed base, before changes | 70 Rust tests + 1 doctest; 76 compiled Python tests pass |
| `cargo fmt --all --check` | Pass |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Pass |
| `cargo test --workspace --locked` (debug, overflow checks on) | **79 tests + 1 doctest pass** |
| `cargo +1.85.0 check --workspace --all-targets --locked` | Pass |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked` | Pass |
| Both Rust examples (`toy`, `solve`) | Pass |
| `python -m maturin build --release --locked`, install resulting wheel, `pytest -q` | **85 pass** |
| `mypy --strict python/sqp_craft` | Pass |
| Python demo, SQP and restoration examples | Pass; HS071 recovers in 8 outer steps with 1 restoration step |
| `--no-default-features` wheel, `pytest tests/test_contract.py` | **42 pass**; native SQP raises `NotImplementedError` |
| Existing 13-case comparison | SQP 13/13 accepted; SLSQP 12/13 |
| Cold/reuse comparison (`--repeats 5`) | Both modes 13/13; all repeated orbit runs accepted |
| Random stress, seeds 1–4, 300 problems per seed, both modes | Both accept 1,186/1,200 with restoration, 970/1,200 without; no false convergence |

The current source adds nine Rust tests and nine collected Python cases. The
original reviewed suite remains intact. Adapter fallback tests deliberately
invalidate an internal matrix pattern after setup or inject rejection after a
real reused solve, so both error routes are exercised deterministically.
Cancellation tests use an engine-iteration hook rather than timing sleeps.

## Reproduce

From a checkout with Rust >=1.85 and CPython >=3.10:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'maturin>=1.9.4,<2' pytest numpy scipy mypy
bash scripts/check.sh
mypy --strict python/sqp_craft
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo +1.85.0 check --workspace --all-targets --locked
python examples/compare_workspace.py --repeats 5 > workspace-results.json
python examples/compare_baseline.py --table
for seed in 1 2 3 4; do
    python examples/stress_random.py --seed "$seed" --count 300 --table
    python examples/stress_random.py --reuse-workspace --seed "$seed" --count 300 --table
done
```

The local native-wheel check used a release build plus explicit install instead
of the script's editable `maturin develop` step. With maturin 1.9.4/PyO3 0.30.1,
`PYO3_BUILD_EXTENSION_MODULE=1` was supplied when building the wheels.
The feature-disabled wheel was separately built with `--no-default-features`,
installed in isolation and tested against the same contract suite.

Raw numerical evidence: [baseline](iteration-3-baseline.json),
[cold/reuse measurements](iteration-3-workspace-results.json),
[stress totals](iteration-3-stress-results.json). Build logs are intentionally
not checked in. The existing CI matrix covers Python 3.10/3.14 on Linux,
macOS and Windows; this delivery extends its baseline job to both workspace
modes. Those remote jobs were not executed locally or on GitHub during this task.

## Limits of the evidence

The orbit measurements are five local repetitions per mode and size, with
alternating order. They show setup savings for these fixtures, not a universal
speedup or a multi-core result. No real target application, free-threaded Python,
or adversarial conditioning campaign was tested. The random family checks
first-order acceptance and does not certify global optimality.
