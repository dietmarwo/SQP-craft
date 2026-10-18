# SQP-craft: thirty eight-thread performance iterations (31–60)

Base: local 0.15.0 commit `7ae0520a9da86fd0b49b9e7e486180ead82a88d8`. Server: eight CPU-equivalents quota, nine visible CPUs. Target is eight runnable numerical workers, not eight replicas of an eight-thread solve.

## Contract

Support both lower latency within one solve and independent-solve throughput. Keep default one-thread behavior, both backends, original callbacks/models, numerical tolerances, objective/KKT gates and the 256-variable NLP cap. Parallelize independent outputs, preserve scalar reduction order where practical, and check every accepted result against the original model. Python callbacks remain owned by their calling thread. A requested width is a maximum concurrency budget, not a promise that all eight workers are useful for every kernel.

Use explicit reusable local execution resources, no mutation of the global Rayon pool. Outer batches must not multiply inner thread pools. Include setup/cold costs separately from warm work, quota/affinity metadata, CPU time, raw samples and failures. Compare single-thread and 8-thread runs, plus 2/4-thread scaling. Keep latency and throughput claims separate. No compilation/stress workers during timings.

Before each iteration: write its detailed plan; then implement, run correctness tests, build release harnesses and measure paired checkpoints. Record negative/noisy results honestly and adapt the next candidate. This is not a guarantee of thirty statistically monotone wall-time improvements. Native parallel work on small matrices can lose; retain serial fallbacks and use batch throughput where appropriate.

## Completed implementation ledger

All 30 iterations are implemented and tested; retained/rejected experiments are documented individually. Defaults, original mathematical inputs and the NLP cap remain. Numerical prototype reversions are recorded rather than hidden. The final release additionally includes all-case three-way validation and repeated baseline/scaling/heldout timing.

| Iteration | Detailed plan | Tests and measurements |
|---:|---|---|
| 31 | [Eight-worker benchmark contract](iteration-31-plan.md) | [Result](iteration-31-results.md) |
| 32 | [Bounded persistent execution resources](iteration-32-plan.md) | [Result](iteration-32-results.md) |
| 33 | [Thread configuration seam](iteration-33-plan.md) | [Result](iteration-33-results.md) |
| 34 | [Parallel inverse-factor columns](iteration-34-plan.md) | [Result](iteration-34-results.md) |
| 35 | [Share compatible owned worker pools](iteration-35-plan.md) | [Result](iteration-35-results.md) |
| 36 | [Larger coupled QP scaling harness](iteration-36-plan.md) | [Result](iteration-36-results.md) |
| 37 | [Parallel BFGS triangle with shared resources](iteration-37-plan.md) | [Result](iteration-37-results.md) |
| 38 | [Parallel BFGS symmetric commit](iteration-38-plan.md) | [Result](iteration-38-results.md) |
| 39 | [Explicit native model thread-budget opt-in](iteration-39-plan.md) | [Result](iteration-39-results.md) |
| 40 | [Parallel independent Householder column updates](iteration-40-plan.md) | [Result](iteration-40-results.md) |
| 41 | [Parallel dense transformed normals](iteration-41-plan.md) | [Result](iteration-41-results.md) |
| 42 | [Parallel KKT projection dots](iteration-42-plan.md) | [Result](iteration-42-results.md) |
| 43 | [Parallel KKT primal reconstruction](iteration-43-plan.md) | [Result](iteration-43-results.md) |
| 44 | [Evidence-based dense scheduling thresholds](iteration-44-plan.md) | [Result](iteration-44-results.md) |
| 45 | [Ordered native independent-solve batch API](iteration-45-plan.md) | [Result](iteration-45-results.md) |
| 46 | [Per-worker solver state reuse](iteration-46-plan.md) | [Result](iteration-46-results.md) |
| 47 | [Linear ordered collection and bounded task claims](iteration-47-plan.md) | [Result](iteration-47-results.md) |
| 48 | [Batch cancellation with input-error precedence](iteration-48-plan.md) | [Result](iteration-48-results.md) |
| 49 | [Compiled Python eight-thread contract coverage](iteration-49-plan.md) | [Result](iteration-49-results.md) |
| 50 | [Parallel bounded dense-row norms](iteration-50-plan.md) | [Result](iteration-50-results.md) |
| 51 | [Disjoint dense Hessian materialization](iteration-51-plan.md) | [Result](iteration-51-results.md) |
| 52 | [Balance inverse-column cost across bounded lanes](iteration-52-plan.md) | [Result](iteration-52-results.md) |
| 53 | [Avoid unused worker creation for small native work](iteration-53-plan.md) | [Result](iteration-53-results.md) |
| 54 | [Parallel native nonlinear propagation workload](iteration-54-plan.md) | [Result](iteration-54-results.md) |
| 55 | [Concurrent caller and cache isolation stress](iteration-55-plan.md) | [Result](iteration-55-results.md) |
| 56 | [Full CPU-budget benchmark matrix and certificates](iteration-56-plan.md) | [Result](iteration-56-results.md) |
| 57 | [Independent three-way numerical audit](iteration-57-plan.md) | [Result](iteration-57-results.md) |
| 58 | [Remove measured overhead from retained production paths](iteration-58-plan.md) | [Result](iteration-58-results.md) |
| 59 | [Fallible factories and fail-closed harness regressions](iteration-59-plan.md) | [Result](iteration-59-results.md) |
| 60 | [Freeze 0.16 and run complete release gates](iteration-60-plan.md) | [Result](iteration-60-results.md) |

Final validation: 196 regular Rust tests, 211 forced all-target tests, 156 Python tests and 52,116 independently journaled solver calls, with unchanged audited statuses/outer iterations/accepted objectives/points. See [RESULTS.md](RESULTS.md), [TABLES.md](TABLES.md), [VALIDITY.md](VALIDITY.md) and [REPRODUCE.md](REPRODUCE.md).
