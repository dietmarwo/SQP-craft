# Iteration 47: results

Ordered batch collection is now linear, with exact one-result-per-index checks, per-worker capacity reservation and bounded dynamic claims. Added a frozen mixed-cost orbit workload whose individual solves retain the original callbacks/reference/KKT assertions; fine claims remain for normal32-job batches.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-47.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 262.107 | 271.327 | 0.966 |
| orbit / 20 / active-set / 1 / 8 / 16 | 102.808 | 102.888 | 0.999 |
| orbit / 40 / active-set / 8 / 1 / 1 | 251.884 | 274.639 | 0.917 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.220 | 17.942 | 0.960 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.

Mixed-cost32-solve confirmation: one worker1877.92 ms versus eight418.32 ms (4.489× throughput); all original orbit20/40 assertions passed. Both wall and CPU samples remain in data/iteration-47-mixed.json.
