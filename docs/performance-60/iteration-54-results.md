# Iteration 54: results

Added a separate nonlinear controlled-ODE propagation model with RK4 forward sensitivities, finer-step known target controls, cached grouped derivatives and explicit native segment parallelism. Old models and pilot limits remain unchanged. Sensitivities pass finite differences/refinement checks, and serial/eight-thread propagated outputs/objectives/gradients are bit-identical.

Full release Rust workspace tests passed before serial paired timings. Each timed child validates every result against unchanged objective/KKT assertions. The process may contain up to eight numerical workers; no build/stress work ran concurrently. One warmup plus three measured samples. Unsupported pre-API configurations remain recorded as failures and are not speedups. Raw samples, commands, hashes, quota and CPU time are in `data/iteration-54.json`.

| Case: family / size / backend / inner threads / workers / jobs | Previous ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| orbit / 40 / active-set / 1 / 1 / 1 | 278.726 | 276.855 | 1.007 |
| orbit / 20 / active-set / 1 / 8 / 16 | 104.742 | 107.090 | 0.978 |
| orbit / 40 / active-set / 8 / 1 / 1 | 254.831 | 262.495 | 0.971 |
| separable / 128 / active-set / 8 / 1 / 1 | 16.293 | 17.412 | 0.936 |

Small changes may be noise. Retained code reduces identifiable work or supplies a tested parallel capability; no per-checkpoint monotonic wall-time claim. Negative/noisy timings and failures are retained.

Development propagation32 scaling (three measured samples after warmup): whole-workflow47.23/29.39/19.85/17.60 ms at1/2/4/8 threads, respectively. All10 outer iterations and objective4.5158650141984815e-13 are identical. Native solve time is also retained separately in each raw job record; target-model construction is included in the whole-workflow number. Raw: data/iteration-54-propagation.json. This is a separate nonlinear integration surrogate.
