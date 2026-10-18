# Iteration 45: results

Added public Rust batch::solve_many with explicit eight-CPU total budget, worker-local factory construction and deterministic indexed results, including independent errors. Non-Send model instances remain supported because construction and use occur on the same worker. Tests cover order, invalid starts and budget rejection.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-45.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 257.891 | 270.305 | 0.954 |
| orbit / 20 / active-set / 1 / 8 / 16 | 97.394 | 94.828 | 1.027 |
| orbit / 40 / active-set / 8 / 1 / 1 | 275.190 | 257.641 | 1.068 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.459 | 19.278 | 0.906 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
