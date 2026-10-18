# Iteration 56: results

The benchmark now pins timed children to exactly eight allowed CPUs, validates CPU-budget products and complete finite/reference/KKT records, preserves failed-run evidence, and separates solver, setup-inclusive wall and aggregate CPU time. Added full1/2/4/8 latency, mixed/balanced batch splits, original/propagation holdouts and larger-QP size selection.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-56.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 275.519 | 292.238 | 0.943 |
| orbit / 20 / active-set / 1 / 8 / 16 | 100.112 | 102.244 | 0.979 |
| orbit / 40 / active-set / 8 / 1 / 1 | 260.669 | 303.782 | 0.858 |
| separable / 128 / active-set / 8 / 1 / 1 | 16.730 | 17.002 | 0.984 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.

The exploratory matrix completed all70 configurations (one warmup and one measured sample each), including both backends, propagation,256-variable separable and mixed batches. These are diagnostic samples, not final precision claims. Source/raw: data/iteration-56-exploratory-matrix.json. The matrix supports flat eight-worker throughput for small NLPs and inner parallelism for expensive native propagation.
