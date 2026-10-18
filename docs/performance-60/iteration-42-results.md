# Iteration 42: results

Independent KKT projection dots now use bounded execution on larger matrices; their original four-lane dot arithmetic and all ordered triangular multiplier solves are unchanged. Complete scalar/parallel c/x/u states match exactly at varied active ranks and dispatch boundaries.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-42.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 257.376 | 274.629 | 0.937 |
| orbit / 20 / active-set / 1 / 8 / 16 | 105.555 | 107.658 | 0.980 |
| orbit / 40 / active-set / 8 / 1 / 1 | 316.654 | 326.267 | 0.971 |
| separable / 128 / active-set / 8 / 1 / 1 | 31.122 | 31.577 | 0.986 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
