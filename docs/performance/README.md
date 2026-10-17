# Ten SQP performance iterations

Version 0.14.0. Base: reviewed upstream main `73d8d0cbab8c8a4fdbebc1cc82e2aa22e2bd509a`.
The numerical source is preserved in local Git checkpoints `sqp-performance-00` through `sqp-performance-10`; the release also includes the final validation safeguards and block-to-full fallback.

## Final paired measurements

CPU: AMD EPYC 9V74 80-Core Processor; Linux x86_64, Rust 1.99.1. One numerical thread. Five measured samples plus one warmup per case/configuration, fresh processes, rotated/reversed order; 120 raw records. No compilation or correctness stress process ran concurrently. No CPU affinity or exclusive-host guarantee. Solver timing excludes fixture construction and process startup. Binary hashes are in the raw JSON.

| Fixture | Baseline full + reuse ms | Final full + reuse ms | Baseline/final | Optional block/hybrid ms | Block-to-full switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| orbit 20 | 267.179 | 242.595 | 1.101x | 218.465 | 1 |
| orbit 40 | 1854.318 | 1857.419 | 0.998x | 2008.982 | 1 |
| separable 128 | 122.577 | 117.055 | 1.047x | 2.131 | 0 |
| restoration 64 | 6.174 | 5.193 | 1.189x | 1.395 | 0 |

Full-mode outer step counts remain 104/167 for the 84/164-variable orbit cases, 21 for the 128-variable separable problem and 4 for the 64-variable restoration problem. Optional blocks use width 4 on the orbit fixture, 2 on separable and 1 on restoration. The hybrid orbit runs use 93/180 outer steps and switch once to full BFGS. Every reported solve passes the same original-unit KKT limits and reference objective checks.

The large orbit full/reuse result is essentially unchanged (0.998x); its optional hybrid mode is about 8.4% slower than baseline full/reuse. Improvements are workload-dependent: no claim is made that each sequential timing sample or every problem gets faster. The ten intermediate reports retain their slower/noisy measurements. The largest gain is on the deliberately separable synthetic model; it must not be represented as a trajectory speedup.

| Fixture | Baseline cold ms | Final cold ms | Baseline/final |
| --- | ---: | ---: | ---: |
| orbit 20 | 297.536 | 282.181 | 1.054x |
| orbit 40 | 2155.214 | 2083.013 | 1.035x |
| separable 128 | 120.709 | 119.568 | 1.010x |
| restoration 64 | 6.693 | 5.648 | 1.185x |

At 164 orbit variables, median BFGS phase time falls from 48.430 to 20.396 ms; conversion from 35.285 to 47.432 ms. The final engine phase is still about 93.1% of wall time. These local component gains do not remove the dense KKT factorization bottleneck.

## Iterations and validation

Each iteration has a pre-implementation plan, code checkpoint, full workspace release tests, warnings-denied Clippy and three measured repetitions after a warmup on the fixed native suite. Later validation caught numerical edge cases and refined the safeguards before release.

| Iteration | Version | Plan | Results |
| --- | --- | --- | --- |
| 1 | 0.5.0 | [Compute one BFGS triangle and mirror it](iteration-01-plan.md) | [measurements](iteration-01-results.md) |
| 2 | 0.6.0 | [Reuse the BFGS workspace](iteration-02-plan.md) | [measurements](iteration-02-results.md) |
| 3 | 0.7.0 | [Maintain the BFGS Cholesky factor with rank updates](iteration-03-plan.md) | [measurements](iteration-03-results.md) |
| 4 | 0.8.0 | [Reuse the SQP step-QP template](iteration-04-plan.md) | [measurements](iteration-04-results.md) |
| 5 | 0.9.0 | [Reuse independent KKT-check buffers](iteration-05-plan.md) | [measurements](iteration-05-results.md) |
| 6 | 0.10.0 | [Recycle derivative callback output buffers](iteration-06-plan.md) | [measurements](iteration-06-results.md) |
| 7 | 0.11.0 | [Preserve sparsity in restoration QPs](iteration-07-plan.md) | [measurements](iteration-07-results.md) |
| 8 | 0.12.0 | [Certify strictly diagonally dominant Hessians in linear storage time](iteration-08-plan.md) | [measurements](iteration-08-results.md) |
| 9 | 0.13.0 | [Add an explicit block-diagonal damped-BFGS mode](iteration-09-plan.md) | [measurements](iteration-09-results.md) |
| 10 | 0.14.0 | [Cache compatible conic conversion data and scatter maps](iteration-10-plan.md) | [measurements](iteration-10-results.md) |

- Final release: 91 Rust tests and one doctest; 97 Python tests; rustfmt, Clippy (workspace/all targets, warnings denied) and mypy passed.
- The 13-case Python reference-objective/original-unit evaluator passes all 26 cold/reuse runs.
- Independent NumPy validation: 1,500 unique random feasible problems (five seeds, 300 each), three settings per release, 9,000 total solves across baseline/final. Both releases converge and pass the independent evaluator on 1,485/1,500 with restoration, in each cold/reuse mode; 13 return restoration_failed and 2 restoration_limit. With restoration disabled, both accept 1,219 and stop at inconsistent linearization on 281. No unexpected exceptions or rejected convergence claims in the release.
- Paired status/acceptance differences: 0. Maximum paired converged objective difference: 0.000e+00. Random cases have no known global optimum; first-order acceptance is the claim.
- Python validation used CPython 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 for the independent evaluator. Linux only; no Windows/macOS or minimum-declared-Rust execution was performed in this session.

## Failure ledger and final refinements

1. The rank-update shortcut initially allowed a finite factor to accompany an explicitly rounded small Hessian that failed the independent SPD gate (seed 4, case 86; three settings). Dimensions <=16 now retain fresh Cholesky checks. The captured input is a permanent regression; the final 1,500-case sweep has no such exceptions. Larger full Hessians retain periodic/fallback factor updates and the adapter SPD gate.
2. Pure block BFGS initially failed both orbit trials. Fresh per-block checks converted that into a controlled numerical stop. A failed block update now consumes one existing reset, switches once to full BFGS and invalidates the QP template; both reference orbit problems then converge under unchanged budgets/tolerances. The switch is reported. Original failed experiments remain in iteration-09/10-blocks.json and final-blocks.json; final-hybrid-blocks.json and the final paired comparison describe the release.
3. Per-iteration wall times are noisy and not monotone. Allocation reductions, matrix-expression counts and phase times explain the intended improvements; the final paired table is the stronger end-to-end evidence. No failed run is counted as a speedup, and no acceptance threshold was weakened.

## Reproduce

```sh
cargo test --workspace --release --locked
cargo build --release -p sqp-craft-solver --example performance --locked
python3 scripts/performance_iteration.py target/release/examples/performance measurements.json
python3 scripts/benchmark_structured.py target/release/examples/performance block-results.json
python -m pip install ".[test,bench]"
python -m pytest -q
python scripts/validate_performance_stress.py stress.json --seed 5 --count 300
```

The archive includes full source, the Python wrapper, raw measurements, validation logs, a checksum manifest and a Git bundle. Restore the checkpoint history with:

```sh
git clone --branch performance-10 history/sqp-performance.bundle SQP-craft-history
```

Build the performance example at `sqp-performance-00` and at `sqp-performance-release` into separate target directories, then pass their binary paths and an output filename to `scripts/compare_performance_final.py`. No benchmark binaries or virtual environments are bundled.

## Scope and next priorities

The application target card remains unfilled. The orbit case is the existing trapezoidal low-thrust fixture, not the unavailable user application. Callbacks, dynamics, transcription, reference objectives, tolerances and declared budgets were kept fixed for comparisons. Full BFGS remains the default; QP workspace reuse is still opt-in. Blocks use existing variable order and are a curvature approximation, not an inferred physical partition.

The solver still has a 256-variable cap, a dense full Hessian, one numerical thread per solve, no second-order correction, no automatic scaling and no industrial robustness guarantee. It reports numerical first-order convergence, not a global or exact certificate. Future application-driven work should target KKT factorization, meaningful curvature partitions or limited-memory representations, and structured QP methods. The current synthetic regression suite remains available when the real trajectory model can be added.

Raw release data: [paired native timings](final-native-comparison.json), [validation summary](validation-summary.json), [Python comparison](final-python-comparison.json). Per-seed baseline/final data and per-iteration measurements are adjacent. Final validation logs are under `validation/`.
