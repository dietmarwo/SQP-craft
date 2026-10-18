# Iteration 49: results

The compiled Python boundary is now exercised at 1/2/4/8 native widths with both backends, owner-thread callbacks/observers, unchanged exception identity, cancellation and independent eight-caller execution. Existing out-of-range guidance is preserved. The full extension suite gates this checkpoint before timing.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-49.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 254.409 | 269.353 | 0.945 |
| orbit / 20 / active-set / 1 / 8 / 16 | 107.965 | 101.803 | 1.061 |
| orbit / 40 / active-set / 8 / 1 / 1 | 259.575 | 260.222 | 0.998 |
| separable / 128 / active-set / 8 / 1 / 1 | 18.399 | 18.701 | 0.984 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
