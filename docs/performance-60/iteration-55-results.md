# Iteration 55: results

Added simultaneous 1/1/2/4-budget native propagation callers, eight-worker independent batches, and 512-variable QP cache-isolation stress. H/rows/bounds/gradients change between calls, with invalid-input and cancellation probes before later successful solves. Every accepted result is independently checked against original KKT and known primal/multiplier targets. Warnings-denied all-target/all-feature Clippy passes.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-55.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 257.875 | 268.723 | 0.960 |
| orbit / 20 / active-set / 1 / 8 / 16 | 112.750 | 100.248 | 1.125 |
| orbit / 40 / active-set / 8 / 1 / 1 | 293.319 | 305.722 | 0.959 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.085 | 19.216 | 0.889 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
