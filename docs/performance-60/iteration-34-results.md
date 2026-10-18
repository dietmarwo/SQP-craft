# Iteration 34: results

Parallel coupled inverse-factor columns are bit-identical to scalar columns, with exact diagonal and n<128 serial fallbacks. The eight-thread native option now performs actual independent active-set factor work while callbacks and serial working-set decisions remain on their original thread.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-34.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 258.463 | 273.183 | 0.946 |
| orbit / 20 / active-set / 1 / 8 / 16 | 119.228 | 112.417 | 1.061 |
| orbit / 40 / active-set / 8 / 1 / 1 | 274.174 | 283.404 | 0.967 |
| separable / 128 / active-set / 8 / 1 / 1 | 19.301 | 21.765 | 0.887 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
