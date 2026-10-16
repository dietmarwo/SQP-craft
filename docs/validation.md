# Validation — iteration 3

Version 0.4.0 · 16 October 2026 UTC. Base: reviewed `main` at
`214f93b0ec0cf407f62fd01ef23df90c7a81fe5c`.
Historical evidence: [iteration 2](iteration-2-validation.md).
Current [results](iteration-3-results.md) and [plan](iteration-3-plan.md).

## Review, local

Linux x86_64 (2-core VM), Rust 1.97.0, CPython 3.13, PyO3 0.30.1,
maturin 1.15.0, pytest 9.1.1, NumPy 2.5.4, SciPy 1.18.1. The pull request's
CI repeats the checks on Linux, macOS and Windows with Python 3.10 and 3.14,
plus Rust 1.85 and the dummy-only build.

| Check | Result |
| --- | --- |
| Delivered iteration 3, `bash scripts/check.sh` | Pass: 79 Rust tests + 1 doctest, 85 Python tests |
| Reviewed, `bash scripts/check.sh` from a clean build (fmt, Clippy `-D warnings`, debug tests, examples) | Pass: **82 Rust tests + 1 doctest, 85 Python tests** |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked` | Pass |
| `mypy --strict python/sqp_craft` | Pass |
| Release wheel in a fresh virtual environment (imports from `site-packages`), `pytest` | **85 pass** |
| `--no-default-features` wheel in a fresh virtual environment, `pytest tests/test_contract.py` | **42 pass**; `method="sqp"` raises `NotImplementedError` |
| New time-limit test against the delivered adapter | Fails (Clarabel's clock exceeds the call's own engine time); passes with the fix |
| `examples/compare_baseline.py` | sqp-craft 13/13 accepted, SLSQP 12/13 |
| `examples/stress_random.py`, seeds 1–4, cold and reuse | Both: 1,186/1,200 with restoration, 970/1,200 without, no false convergence |
| `examples/compare_workspace.py --repeats 1` | All cases and timed orbit runs accepted |
| `examples/restore_hs071.py` | Converges in 8 iterations, 1 restoration step |
| Delivered vs reviewed release builds, orbit cases, alternating and balanced orders | Bit-identical objectives and iteration counts; conversion time about 0.145 → 0.075 s (164 variables); engine time within the ±5% code-layout variation (a no-op build moved it 3–4%) |

The three new Rust tests are:

- the direct CSC conversion equals the sorted triplet conversion bit for bit
  on 300 random models;
- Clarabel's time-limit clock on a reused workspace counts only the current
  call, over 300 reused solves;
- the entry-wise BFGS update is bit-identical to the matrix expression, at
  sizes 1–64 with damping.

The option measurements in the results (the `faer-sparse` factorizer,
refinement off, `qp_tolerance = 1e-8`) used throwaway builds and Clarabel's
internal timers printed after each solve. They are not part of the shipped
code.

## Delivered record

The delivery's own run used Rust 1.99.1 (and 1.85 for the minimum-version
check), CPython 3.12.14, maturin 1.9.4 (with `PYO3_BUILD_EXTENSION_MODULE=1`)
and the same NumPy/SciPy. It reported 79 Rust tests + 1 doctest, 85 Python
tests and 42 dummy-only contract tests. The JSON files
[baseline](iteration-3-baseline.json),
[cold/reuse measurements](iteration-3-workspace-results.json) and
[stress totals](iteration-3-stress-results.json) are from that run and host.
The review changes leave every trajectory bit-identical, so their outcomes
still hold; their timings are not re-measured on the delivery's host.

## Reproduce

From a checkout with Rust >=1.85 and CPython >=3.10:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install maturin pytest numpy scipy mypy
bash scripts/check.sh
mypy --strict python/sqp_craft
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
python examples/compare_workspace.py --repeats 5 > workspace-results.json
python examples/compare_baseline.py --table
for seed in 1 2 3 4; do
    python examples/stress_random.py --seed "$seed" --count 300 --table
    python examples/stress_random.py --reuse-workspace --seed "$seed" --count 300 --table
done
```

Build logs are intentionally not checked in.

## Limits of the evidence

The orbit measurements are local repetitions with alternating order on a
shared VM, where wall time varied by about ±5%. Different builds of the same
engine code also differ by up to about 5% in engine time, from code layout
alone. They show setup and
conversion savings for these fixtures, not a universal speedup or a
multi-core result. No real target application, free-threaded Python or
adversarial conditioning campaign was tested. The random family checks
first-order acceptance and does not certify global optimality.
