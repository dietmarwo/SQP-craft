# Iteration 36: results

Added independent planted coupled-QP scaling coverage at 128/256/512/1024 variables, fresh Hessians on every call, retained resources, and explicit resource setup/CPU-time records. Every result must pass the original KKT rule and known primal/dual solution. The existing NLP cap and all original callbacks remain unchanged.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-36.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 284.186 | 280.453 | 1.013 |
| orbit / 20 / active-set / 1 / 8 / 16 | 102.836 | 100.844 | 1.020 |
| orbit / 40 / active-set / 8 / 1 / 1 | 367.696 | 271.983 | 1.352 |
| separable / 128 / active-set / 8 / 1 / 1 | 20.384 | 33.312 | 0.612 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.

## Fresh-H coupled QP scaling

Three solves per sample, separate resource-setup measurement; all original-model KKT/planted primal/dual checks passed at all widths. Numerical time (not whole-process time):

- n=128: 1t 1.39ms (1.00x), 2t 1.66ms (0.84x), 4t 2.74ms (0.51x), 8t 1.43ms (0.97x)
- n=256: 1t 7.17ms (1.00x), 2t 7.11ms (1.01x), 4t 6.66ms (1.08x), 8t 7.64ms (0.94x)
- n=512: 1t 55.34ms (1.00x), 2t 37.54ms (1.47x), 4t 38.01ms (1.46x), 8t 29.65ms (1.87x)
- n=1024: 1t 398.90ms (1.00x), 2t 330.17ms (1.21x), 4t 263.59ms (1.51x), 8t 228.55ms (1.75x)

Raw: data/iteration-36-qp-scaling.json.gz. Larger QPs demonstrate useful inverse parallelism while the small-size losses remain visible.
