# Iteration 44: results

Raised dense-QP dispatch to 512 variables and BFGS triangle/commit dispatch to 256, following measured small-size losses and 512/1024 gains. Expanded exact parallel tests above the new boundaries. This scheduling change retains scalar formulas, original models, budgets and residual thresholds.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-44.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 270.881 | 276.611 | 0.979 |
| orbit / 20 / active-set / 1 / 8 / 16 | 99.180 | 106.710 | 0.929 |
| orbit / 40 / active-set / 8 / 1 / 1 | 306.666 | 272.005 | 1.127 |
| separable / 128 / active-set / 8 / 1 / 1 | 26.368 | 18.854 | 1.399 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.

## Threshold confirmation

Five-repeat separable256 comparison: eight-thread time falls from 546.50 to 115.80 ms after avoiding fine-grained QP dispatch; scalar candidate 107.95 ms. The remaining eight-thread overhead is visible. Full 1/2/4/8 fresh-H QP scaling is in data/iteration-44-qp-scaling.json.gz; threshold comparison raw samples are in data/iteration-44-threshold-confirmation.json.
