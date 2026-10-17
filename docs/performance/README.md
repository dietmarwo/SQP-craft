# Ten SQP performance iterations

Version 0.14.0. Base: reviewed upstream main `73d8d0cbab8c8a4fdbebc1cc82e2aa22e2bd509a`.
The delivered archive preserved the numerical source as local Git checkpoints `sqp-performance-00` through `sqp-performance-10` in a bundle, which is not part of this repository. The release also includes the final validation safeguards and block-to-full fallback.

## Review of the performance release

The delivery built and passed its 91 Rust tests, doctest and 97 Python tests. The review rebuilt the baseline checkpoint (`main` plus the benchmark), the delivered release and the reviewed code on a 2-core Linux VM (Rust 1.97.0). It reproduced the paired benchmark and checked every result against `main`.

**One regression: an indefinite B aborted the solve above 16 variables.** Iteration 3 replaced the fresh Cholesky check of each BFGS update with a rank-updated factor. That factor tracks a nearby matrix, not the rounded B that the QP receives. When cancellation in the explicit rank-two terms left B indefinite, the factor still looked valid, the QP adapter rejected B with `Unsupported`, and the whole solve failed. On `main` the same B was caught at the update and reset. The delivery saw this at n = 2 (seed 4, case 86) and kept fresh checks for n ≤ 16 only. Padded with 20 decoupled variables, the same problem fails on the delivered release in three different weightings and converges on `main` (`crates/sqp/tests/hessian_gate.rs`, `tests/test_performance_options.py`).

**Fix, and one positive-definiteness check per iteration instead of two.** The adapter already ran an exact check on every B it received. It now returns `QpOutcome::NotPositiveDefinite` instead of an error and keeps its cached workspace and conversion data. The driver then resets B and re-solves, as after a failed update, and counts the event in `RunReport.qp_not_positive_definite` (also in Python). `QpBackend::solve` documents this as the contract for every backend that needs a positive-definite Hessian, and scripted-backend tests pin the driver's handling: one rejection resets B and re-solves the same QP, and repeated rejections stop with `numerical_failure` once the reset budget is spent. Full-mode BFGS therefore no longer factorizes at all: the factor, the rank updates, the 32-update refresh and the n ≤ 16 rule are removed. Explicit blocks keep their exact per-block check, so a failed block still switches to full BFGS at the update.

**Conversion back to baseline cost.** The adapter's check allocated a fresh 215 KB dense matrix per QP at n = 164, above glibc's mmap threshold, and ran the dominance sums as a separate pass. On dense BFGS matrices those sums never succeed. The check now reuses one buffer, folds the sums into the copy, and stops them at the first row that fails.

| Orbit, 164 variables, reuse; medians of 4 rotating rounds | Baseline | Delivered | Reviewed |
| --- | ---: | ---: | ---: |
| BFGS phase (ms) | 95.6 | 47.1 | **33.8** |
| Conversion including the positive-definiteness check (ms) | 68.3 | 83.4 | **71.4** |
| BFGS + conversion (ms) | 163.9 | 130.5 | **105.2** |
| Clarabel engine (ms), same 1,444 interior-point iterations | 2,640 | 2,654 | 2,784 |
| Wall (ms) | 2,910 | 2,881 | 2,983 |

The engine receives bit-identical input in all three builds, and its code is unchanged. Its time still moves by about ±5% with code layout. An earlier reviewed build, which differed only in the dominance loop, ran the same engine work in 2,631 ms and had the fastest wall time of the three (2,837 ms). Wall-time differences of a few percent between builds therefore do not measure the work; the BFGS and conversion phases do. At 84 variables BFGS + conversion is 18.6 / 16.5 / 11.8 ms. On the restoration fixture it is 0.50 / 0.29 / 0.26 ms. The delivered block modes are unchanged (same step counts and times).

Results are bit-identical to `main` in full mode, in status, objective bits and iteration counts:

- all 6,000 stress runs (seeds 1–5 × 300 problems, restoration on and off, cold and reuse);
- all 26 runs of the 13-case comparison, including interior-point iteration counts;
- the orbit objectives.

The delivered release gives the same results on all of these as well. Fat LTO and thin LTO were measured and did not help (thin LTO with 16 codegen units was 7% slower), so the release profile is unchanged.

Local validation of the review (Rust 1.97.0, CPython 3.13, maturin 1.15.0):

- `scripts/check.sh` from a clean build passes: fmt, Clippy `-D warnings`, 96 Rust tests + 1 doctest in debug, the examples, and 98 Python tests.
- `cargo doc` with `-D warnings` and `mypy --strict` both pass.
- A fresh-venv release wheel passes all 98 tests.
- The dummy-only wheel passes the 42 contract tests.

The new tests:

- the padded regression in Rust and Python, which fails on the delivered release;
- the driver's handling of a scripted backend that rejects one QP, or every QP after the first;
- the adapter's outcome and cache retention on a non-positive-definite Hessian;
- the check's reused buffer across dimension and pattern changes;
- the exact per-block check.

Repository changes:

- The 31 build and test logs under `validation/` and the ten 424 KB per-seed stress dumps are dropped; build logs stay out of the repository by policy, and `validation-summary.json` keeps the totals.
- `scripts/finalize_performance_iteration.py` is dropped. It was the delivery's local workflow (it bumps versions, copies logs into `docs/`, commits, tags and writes outside the repository).
- `compare_performance_final.py` becomes `compare_performance.py`, which pairs any number of labeled builds.
- The other measurement scripts and `tests/test_performance_options.py` are rewritten readably with the same behavior.

## Final paired measurements (delivered, on the delivery's host)

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

1. The rank-update shortcut initially allowed a finite factor to accompany an explicitly rounded small Hessian that failed the independent SPD gate (seed 4, case 86; three settings). Dimensions <=16 now retain fresh Cholesky checks. The captured input is a permanent regression; the final 1,500-case sweep has no such exceptions. Larger full Hessians retain periodic/fallback factor updates and the adapter SPD gate. *Review:* the same case padded to n = 22 still failed on the release; the review removed the rank-update factor and made the adapter's exact check the gate (see above).
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

To pair builds, build the `performance` example of each into its own target directory and run
`python3 scripts/compare_performance.py out.json base=PATH_A new=PATH_B --blocks new --repeats 5`.
No benchmark binaries or virtual environments are kept in the repository.

## Scope and next priorities

The application target card remains unfilled. The orbit case is the existing trapezoidal low-thrust fixture, not the unavailable user application. Callbacks, dynamics, transcription, reference objectives, tolerances and declared budgets were kept fixed for comparisons. Full BFGS remains the default; QP workspace reuse is still opt-in. Blocks use existing variable order and are a curvature approximation, not an inferred physical partition.

The solver still has a 256-variable cap, a dense full Hessian, one numerical thread per solve, no second-order correction, no automatic scaling and no industrial robustness guarantee. It reports numerical first-order convergence, not a global or exact certificate. Future application-driven work should target KKT factorization, meaningful curvature partitions or limited-memory representations, and structured QP methods. The current synthetic regression suite remains available when the real trajectory model can be added.

Raw delivered data: [paired native timings](final-native-comparison.json), [validation summary](validation-summary.json), [Python comparison](final-python-comparison.json); per-iteration measurements are adjacent. The per-seed stress dumps and build logs were dropped in the review; `scripts/validate_performance_stress.py` regenerates the former.
