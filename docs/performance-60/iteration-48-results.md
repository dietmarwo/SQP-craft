# Iteration 48: results

Batch cancellation now avoids constructing skipped models, preserves completed indexed results, and keeps malformed-start errors ahead of cancellation. Pre-cancelled batches avoid worker creation. Tests verify deterministic mid-batch stop, zero later factories and original error precedence; running callbacks remain cooperatively cancellable only.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-48.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 250.957 | 251.709 | 0.997 |
| orbit / 20 / active-set / 1 / 8 / 16 | 96.798 | 97.759 | 0.990 |
| orbit / 40 / active-set / 8 / 1 / 1 | 263.267 | 254.430 | 1.035 |
| separable / 128 / active-set / 8 / 1 / 1 | 18.498 | 17.334 | 1.067 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
