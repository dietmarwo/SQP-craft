# Iteration 39: results

Native models can explicitly receive and use the worker budget while all callback entry points stay on their caller thread. A deliberately non-Send Rc-backed model passes with both QP backends at 1/4/8 threads and invalid input prevents configuration/evaluation. Existing models inherit a serial no-op hook.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-39.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 266.965 | 265.276 | 1.006 |
| orbit / 20 / active-set / 1 / 8 / 16 | 102.489 | 101.491 | 1.010 |
| orbit / 40 / active-set / 8 / 1 / 1 | 328.138 | 314.166 | 1.044 |
| separable / 128 / active-set / 8 / 1 / 1 | 25.104 | 26.229 | 0.957 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
