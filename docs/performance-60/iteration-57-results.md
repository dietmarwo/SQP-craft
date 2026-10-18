# Iteration 57: results

Added a fresh-process baseline/scalar/eight-budget audit around unchanged model generators. Every actual solve is journaled and independently checked with its own multipliers, including full variant objectives/points. Completeness, versions/extension hashes, original source hashes and paired outcomes are gated. A2760-call quick campaign passes before timing, with larger inverse/failed-factor recovery tests above both scheduling boundaries.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-57.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 282.950 | 278.417 | 1.016 |
| orbit / 20 / active-set / 1 / 8 / 16 | 131.400 | 133.302 | 0.986 |
| orbit / 40 / active-set / 8 / 1 / 1 | 276.370 | 285.114 | 0.969 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.947 | 17.823 | 1.007 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
