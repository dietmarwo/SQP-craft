# SQP-craft: thirty eight-thread performance iterations (31–60)

Base: local 0.15.0 commit `7ae0520a9da86fd0b49b9e7e486180ead82a88d8`. Server: eight CPU-equivalents quota, nine visible CPUs. Target is eight runnable numerical workers, not eight replicas of an eight-thread solve.

## Contract

Support both lower latency within one solve and independent-solve throughput. Keep default one-thread behavior, both backends, original callbacks/models, numerical tolerances, objective/KKT gates and the 256-variable NLP cap. Parallelize independent outputs, preserve scalar reduction order where practical, and check every accepted result against the original model. Python callbacks remain owned by their calling thread. A requested width is a maximum concurrency budget, not a promise that all eight workers are useful for every kernel.

Use explicit reusable local execution resources, no mutation of the global Rayon pool. Outer batches must not multiply inner thread pools. Include setup/cold costs separately from warm work, quota/affinity metadata, CPU time, raw samples and failures. Compare single-thread and 8-thread runs, plus 2/4-thread scaling. Keep latency and throughput claims separate. No compilation/stress workers during timings.

Before each iteration: write its detailed plan; then implement, run correctness tests, build release harnesses and measure paired checkpoints. Record negative/noisy results honestly and adapt the next candidate. This is not a guarantee of thirty statistically monotone wall-time improvements. Native parallel work on small matrices can lose; retain serial fallbacks and use batch throughput where appropriate.

## Planned sequence

| Iteration | Planned work |
|---:|---|
| 31 | [Eight-worker benchmark contract](iteration-31-plan.md): Add common 1/2/4/8 CPU-budget native latency and batch-throughput fixtures; immutable 0.15 reference; verify every result with existing objective/KKT assertions. |
| 32 | [Bounded persistent execution resources](iteration-32-plan.md): Introduce an isolated explicit-width Rayon pool with sequential default and ordered disjoint output; test exact worker bound and panic propagation. |
| 33 | [Thread configuration seam](iteration-33-plan.md): Allow explicit native thread budgets and provide a backend capability seam; retain explicit errors for unsupported backends and dummy methods. |
| 34 | [Parallel inverse-factor columns](iteration-34-plan.md): Parallelize independent triangular inverse columns without changing within-column reductions; verify bit-identical factors. |
| 35 | [Parallel BFGS triangle](iteration-35-plan.md): Assign independent columns of the rank-two update to workers; preserve finite rejection and all scalar operation orders. |
| 36 | [Parallel BFGS mirror](iteration-36-plan.md): Write independent target columns from immutable computed lower entries, retaining exact symmetry and rejected-update behavior. |
| 37 | [Parallel BFGS matrix-vector work](iteration-37-plan.md): Parallelize independent output rows only above measured thresholds; retain sequential reduction order or reject the candidate. |
| 38 | [Parallel Householder column updates](iteration-38-plan.md): Run independent reflector target columns concurrently after the unchanged ordered normal accumulation. |
| 39 | [Parallel transformed constraint normals](iteration-39-plan.md): Compute independent projected normal components concurrently for sufficiently dense/large rows. |
| 40 | [Parallel KKT projection dots](iteration-40-plan.md): Distribute independent J-column dot products, preserving each scalar dot reduction and working-set decisions. |
| 41 | [Parallel KKT primal reconstruction](iteration-41-plan.md): Assign independent primal coordinates with original column accumulation order; retain multiplier solves serially. |
| 42 | [Dense QP scheduling thresholds](iteration-42-plan.md): Measure dispatch versus work on 128/256/512/1024 QPs and retain thresholds that avoid small-work regressions. |
| 43 | [Pool reuse across execution owners](iteration-43-plan.md): Share compatible owned pool handles without mutating the global Rayon pool; test teardown, width changes and concurrent callers. |
| 44 | [Nested execution budget](iteration-44-plan.md): Ensure an outer batch reuses its worker budget and avoids a second worker pool per inner solve. |
| 45 | [Ordered native batch entry point](iteration-45-plan.md): Add a usable Rust factory-based batch API with independent solver/model state and deterministic result ordering. |
| 46 | [Per-worker solver reuse](iteration-46-plan.md): Retain each worker backend across tasks so compatible buffers and exact-key factors can be reused without cross-model state leaks. |
| 47 | [Dynamic batch scheduling](iteration-47-plan.md): Use work stealing or atomic next-task assignment to balance heterogeneous solves, retaining deterministic output indices. |
| 48 | [Batch cancellation and errors](iteration-48-plan.md): Bound cancellation latency between tasks, retain completed results and explicit per-task failures; never turn failures into throughput wins. |
| 49 | [Python eight-thread API coverage](iteration-49-plan.md): Exercise native threads=8 through the real extension, including callback ownership, exceptions, observer and cancellation semantics. |
| 50 | [Dense-row QP norm work](iteration-50-plan.md): Distribute expensive independent row norm work while leaving small sparse rows serial. |
| 51 | [Hessian materialization work](iteration-51-plan.md): Parallelize disjoint dense factor-input columns where measured work justifies dispatch; preserve cache invalidation. |
| 52 | [Large unconstrained QP coverage](iteration-52-plan.md): Add coupled larger QPs and measured independent-column scaling with checked residuals; retain the 256-variable NLP pilot cap. |
| 53 | [BFGS task partition tuning](iteration-53-plan.md): Balance triangular column work by estimated arithmetic rather than equal column counts; compare numerical identity and latency. |
| 54 | [Short native-work bypass](iteration-54-plan.md): Avoid pool entry for small matrices, sparse normals and zero work; test threshold boundaries. |
| 55 | [Eight-worker callback workload](iteration-55-plan.md): Add a costly deterministic Rust model to distinguish callback throughput from tiny-kernel overhead without changing old models. |
| 56 | [Concurrency isolation stress](iteration-56-plan.md): Test simultaneous callers, different thread budgets, repeated and changed Hessians, independent errors and cancellation. |
| 57 | [Full CPU-budget benchmark matrix](iteration-57-plan.md): Retain latency, throughput, CPU time and cold/warm measurements for 1/2/4/8 configurations and both backends. |
| 58 | [Adversarial parallel numerical audit](iteration-58-plan.md): Pair serial/eight-thread outcomes on fixed random QPs/NLPs, elastic/restoration/options and independent KKT checks. |
| 59 | [Final measured scheduling refinement](iteration-59-plan.md): Apply only evidence-backed scheduling or allocation refinements discovered by the full CPU-budget matrix; record rejected experiments. |
| 60 | [Release integration and reproduction](iteration-60-plan.md): Freeze 0.16 source, run full Rust/Python/style/docs gates, verify baseline patch/history and package all thirty plans/results/raw evidence. |

Candidates may change after measurement. Each result file records what actually shipped and why. Final source will be delivered as a new version; the 0.15 package remains intact.
