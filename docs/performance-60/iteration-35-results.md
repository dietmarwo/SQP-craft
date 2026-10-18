# Iteration 35: results

Compatible execution owners now share a weakly indexed owned pool, avoiding duplicate QP/BFGS pools in the next kernels. Pool creation is serialized briefly; kernel work never holds the registry lock. Different widths and owner teardown are tested. No per-fixture latency gain is attributed to this resource-sharing checkpoint.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-35.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 265.420 | 263.747 | 1.006 |
| orbit / 20 / active-set / 1 / 8 / 16 | 89.749 | 100.245 | 0.895 |
| orbit / 40 / active-set / 8 / 1 / 1 | 276.279 | 261.697 | 1.056 |
| separable / 128 / active-set / 8 / 1 / 1 | 20.019 | 18.997 | 1.054 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
