# Iteration 58: results

Removed the small-pilot parallel BFGS prototype after measured losses, restoring the exact tested 0.15 scalar implementation and avoiding its driver pool entirely. Removed cyclic inverse-lane allocation after no reliable benefit; large independent-column QP work keeps its generic bounded parallel schedule. Original prototype tags/raw evidence remain, and a public 256-variable scalar/eight-thread budget solve now matches exactly.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-58.json.gz`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 268.471 | 267.279 | 1.004 |
| orbit / 20 / active-set / 1 / 8 / 16 | 100.411 | 99.386 | 1.010 |
| orbit / 40 / active-set / 8 / 1 / 1 | 279.188 | 266.643 | 1.047 |
| separable / 128 / active-set / 8 / 1 / 1 | 17.754 | 17.010 | 1.044 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.

Seven-repeat 256-variable confirmation: eight-thread budget time 118.26→111.74 ms and aggregate CPU 135.32→113.86 ms after removing the prototype. Scalar samples 112.06→117.11 ms show remaining host/code-layout variability; no scalar win is claimed. Both paths now use byte-identical0.15 BFGS. Raw: data/iteration-58-confirmation.json.gz.
