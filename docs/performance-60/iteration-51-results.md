# Iteration 51: results

Changed fully stored dense lower Hessians now materialize disjoint factor-input columns in parallel at 512+ variables. Scalar sparse loading and all complete bit-key/cache invalidation gates remain unchanged. Initial loads, factor reuse and changed-value invalidation match scalar buffers/factors exactly.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-51.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 260.007 | 285.772 | 0.910 |
| orbit / 20 / active-set / 1 / 8 / 16 | 103.580 | 98.636 | 1.050 |
| orbit / 40 / active-set / 8 / 1 / 1 | 265.989 | 246.876 | 1.077 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.636 | 18.141 | 0.972 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
