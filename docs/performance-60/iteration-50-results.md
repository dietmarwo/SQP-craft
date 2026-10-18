# Iteration 50: results

Large bounded dense-row norm preparation now distributes independent rows, preserving the original scalar norm reduction. Free rows remain zero and do not count toward dispatch work; sparse/small fixtures keep the original serial iterator. A1024-row dense/free mixture matches scalar output exactly.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-50.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 272.834 | 264.643 | 1.031 |
| orbit / 20 / active-set / 1 / 8 / 16 | 96.070 | 99.984 | 0.961 |
| orbit / 40 / active-set / 8 / 1 / 1 | 271.435 | 252.918 | 1.073 |
| separable / 128 / active-set / 8 / 1 / 1 | 20.166 | 18.103 | 1.114 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
