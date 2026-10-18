# Iteration 38: results

Added alias-free parallel symmetric commit from the completed lower triangle, with all finite/SPD gates before modifying B. Scalar default mirror/swap remains, and exact serial/eight-thread rank-two update sequences continue to match bit-for-bit. This copies extra lower entries and its dispatch threshold remains subject to final scaling evidence.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-38.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 263.431 | 281.781 | 0.935 |
| orbit / 20 / active-set / 1 / 8 / 16 | 95.414 | 99.802 | 0.956 |
| orbit / 40 / active-set / 8 / 1 / 1 | 283.663 | 317.613 | 0.893 |
| separable / 128 / active-set / 8 / 1 / 1 | 24.754 | 25.284 | 0.979 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
