# Iteration 33: results

Options.threads now accepts 1..8 for native methods and propagates an explicit backend worker budget. Dummy/custom unsupported backends retain explicit errors. Resources are owned and retained per backend; numerical kernels are still scalar at this configuration checkpoint. Restored the original libc pin; existing numerical dependency versions remain unchanged.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-33.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 273.918 | 254.689 | 1.075 |
| orbit / 20 / active-set / 1 / 8 / 16 | 93.509 | 99.795 | 0.937 |
| orbit / 20 / active-set / 1 / 1 / 16 | 479.886 | 490.640 | 0.978 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
