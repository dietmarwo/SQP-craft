# Iteration 52: results

The cyclic cost-balancing experiment was revised after two 7-repeat campaigns showed a 1024-variable regression (~5–10%). Retain bounded cyclic lanes only below 768 variables and the original cache-contiguous dynamic schedule for larger factors. The original candidate, raw timing and source experiment tag remain preserved; all within-column arithmetic and exact identity checks are unchanged.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-52.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 266.827 | 259.232 | 1.029 |
| orbit / 20 / active-set / 1 / 8 / 16 | 99.393 | 98.475 | 1.009 |
| orbit / 40 / active-set / 8 / 1 / 1 | 251.998 | 253.110 | 0.996 |
| separable / 128 / active-set / 8 / 1 / 1 | 21.745 | 18.121 | 1.200 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.

The initial cyclic candidate is preserved as source tag sqp-parallel-52-cyclic-experiment and data/iteration-52-cyclic.json.gz. Two seven-repeat comparisons showed the 1024 regression; the retained hybrid keeps the generic schedule from 768 upward. A further five-repeat retained comparison was effectively flat at 512 and about 3.4% slower at 1024, so no robust wall-time gain is claimed for this scheduler experiment. All samples are retained in iteration-52-*-qp/recheck/confirmation JSON files. Final scheduling will be reassessed after broader scaling.
