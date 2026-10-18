# Iteration 32: results

Added explicit reusable Rayon resources, deterministic disjoint-output operations and bounded ordered task maps. The benchmark now exercises this production execution resource, including cold creation/teardown in batch wall time. Core remains dependency-free; Rayon global resources are untouched.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-32.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 282.324 | 282.155 | 1.001 |
| orbit / 20 / active-set / 1 / 8 / 16 | 100.184 | 110.967 | 0.903 |
| orbit / 20 / active-set / 1 / 1 / 16 | 500.586 | 484.010 | 1.034 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.
