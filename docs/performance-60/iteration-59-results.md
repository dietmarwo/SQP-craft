# Iteration 59: results

Added fallible worker-local factories with ordered per-job errors and retained successful backend reuse. Meaningful subprocess tests reject nonzero exits, invalid/nonfinite certificates and excessive budgets before reporting speedups, and verify a valid protocol/warmup split. API/README now describe the actual retained implementation, serial small-pilot BFGS/Clarabel engine, model opt-in and eight-thread budget batches. Full rebuilt Python and warnings-denied Clippy gates pass before timing.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-59.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 265.233 | 267.380 | 0.992 |
| orbit / 20 / active-set / 1 / 8 / 16 | 97.654 | 96.510 | 1.012 |
| orbit / 40 / active-set / 8 / 1 / 1 | 261.151 | 279.536 | 0.934 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.160 | 17.442 | 0.984 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
