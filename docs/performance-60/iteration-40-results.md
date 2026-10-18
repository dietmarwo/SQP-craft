# Iteration 40: results

Householder target columns may now update concurrently after the original ordered accumulator is complete. Exact support, full norm, scalar axpy and sign correction remain unchanged. Serial/eight-thread reflector state is tested exactly across dispatch boundaries; default and small n<256 work stays scalar.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-40.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 282.643 | 283.532 | 0.997 |
| orbit / 20 / active-set / 1 / 8 / 16 | 98.931 | 105.897 | 0.934 |
| orbit / 40 / active-set / 8 / 1 / 1 | 304.046 | 308.810 | 0.985 |
| separable / 128 / active-set / 8 / 1 / 1 | 23.245 | 22.529 | 1.032 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
