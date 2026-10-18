# Iteration 60: results

Frozen local0.16.0 with explicit native/batch eight-worker support, retained useful large-QP/propagation parallelism and measured scalar fallbacks. Worker setup is included in the cooperative solve deadline and skipped on pre-cancel. Full release/debug/all-target/all-feature Rust, compiled Python, Clippy/rustfmt/rustdoc,52,116-call independent numerical audit and1/8-budget harder-QP census gate this release before timing.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-60.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 260.255 | 254.292 | 1.023 |
| orbit / 20 / active-set / 1 / 8 / 16 | 102.889 | 96.401 | 1.067 |
| orbit / 40 / active-set / 8 / 1 / 1 | 254.466 | 260.505 | 0.977 |
| separable / 128 / active-set / 8 / 1 / 1 | 16.557 | 16.458 | 1.006 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
