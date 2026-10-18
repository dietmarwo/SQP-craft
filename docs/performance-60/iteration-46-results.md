# Iteration 46: results

Batch execution now creates solver state once per worker and dynamically assigns jobs, preserving model owner-thread affinity and ordered per-job outputs. Production solve_many and the native benchmark reuse backend buffers/exact caches across independent starts; BFGS and warm active sets still restart per solve. All completed solves retain original objective/KKT checks.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-46.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 264.706 | 257.197 | 1.029 |
| orbit / 20 / active-set / 1 / 8 / 16 | 97.008 | 95.573 | 1.015 |
| orbit / 40 / active-set / 8 / 1 / 1 | 259.569 | 251.871 | 1.031 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.735 | 17.302 | 1.025 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
