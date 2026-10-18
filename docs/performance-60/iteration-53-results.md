# Iteration 53: results

Small native work now validates/stores its eight-worker maximum without starting idle QP/curvature workers. Larger QPs and explicit native model parallelism still acquire shared resources when useful. Clarabel engine remains serial; existing callbacks/defaults remain unchanged. Added direct small-QP no-worker ownership checks and preserved cold QP resource setup as a separate measurement.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-53.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 262.375 | 267.705 | 0.980 |
| orbit / 20 / active-set / 1 / 8 / 16 | 102.637 | 100.367 | 1.023 |
| orbit / 40 / active-set / 8 / 1 / 1 | 265.370 | 277.648 | 0.956 |
| separable / 128 / active-set / 8 / 1 / 1 | 18.562 | 19.512 | 0.951 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
