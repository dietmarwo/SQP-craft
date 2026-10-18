# Iteration 41: results

Dense transformed constraint normals now distribute independent dot outputs across the shared worker budget. Variable/singleton/small normals retain scalar paths. Exact scalar/parallel tests cover both signs and 128/256 boundaries; original reflector references are test-only to keep the production library warning-free.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-41.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 283.390 | 273.179 | 1.037 |
| orbit / 20 / active-set / 1 / 8 / 16 | 100.010 | 115.202 | 0.868 |
| orbit / 40 / active-set / 8 / 1 / 1 | 318.060 | 288.997 | 1.101 |
| separable / 128 / active-set / 8 / 1 / 1 | 29.483 | 27.742 | 1.063 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
