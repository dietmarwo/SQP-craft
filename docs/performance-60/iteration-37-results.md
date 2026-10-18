# Iteration 37: results

BFGS triangle columns now use shared bounded execution resources with unchanged scalar expressions and finite rejection. Default one-thread path is preserved, and eight-thread exact/damped update sequences are tested bit-for-bit. QP and curvature owners share one compatible worker pool.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-37.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 262.426 | 252.888 | 1.038 |
| orbit / 20 / active-set / 1 / 8 / 16 | 96.936 | 89.950 | 1.078 |
| orbit / 40 / active-set / 8 / 1 / 1 | 278.951 | 289.314 | 0.964 |
| separable / 128 / active-set / 8 / 1 / 1 | 20.063 | 23.887 | 0.840 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
