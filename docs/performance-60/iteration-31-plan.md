# Iteration 31: Eight-worker benchmark contract

## Implementation

Add a separate common Rust example derived from the unchanged 0.15 native benchmark, preserving its callbacks and mathematical assertions. Expose inner threads, outer worker count and job count. Run fixed jobs with one or eight OS workers; construct and own each model inside its worker. Collect ordered per-job records and record wall/CPU time and completed job count. Add a Python paired driver that records command lines, all samples, hashes, CPU quota/affinity and rejects failures.

## Correctness and performance checks

Compile/test the full release workspace before timing. Every solve must pass the original reference objective and original-unit KKT thresholds. Freeze primary latency sizes orbit20/40, separable128 and restoration64; holdouts orbit30/48, separable96/192 and restoration48/96. Batch comparison uses the same total jobs and single-thread inner solves at worker counts1/8. Freeze independent larger coupled QPs for later kernel scaling, never conflate them with old NLPs. No parallel numerical code changes in this baseline iteration.

## Risks

Eight concurrent jobs may be quota-throttled or affected by host load. Time total work and report aggregate throughput, not summed per-solve durations; do not claim an eightfold speedup.
