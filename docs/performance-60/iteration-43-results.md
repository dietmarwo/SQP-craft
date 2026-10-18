# Iteration 43: results

KKT primal reconstruction now assigns disjoint coordinate tiles, preserving original column accumulation and exact zero skips without atomics or a parallel floating-point reduction. Complete c/x/u identity tests remain passing; triangular multiplier solves and working-set decisions remain ordered.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-43.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 279.254 | 263.389 | 1.060 |
| orbit / 20 / active-set / 1 / 8 / 16 | 97.475 | 104.902 | 0.929 |
| orbit / 40 / active-set / 8 / 1 / 1 | 322.342 | 324.910 | 0.992 |
| separable / 128 / active-set / 8 / 1 / 1 | 27.372 | 26.632 | 1.028 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
