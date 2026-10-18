# SQP-craft 0.16: thirty eight-thread performance iterations

Completed iterations 31–60 on tested local 0.15 (`7ae0520a9da86fd0b49b9e7e486180ead82a88d8`). Every iteration has a pre-implementation plan, implemented experiment/capability and measured comparisons. All regular release-test results are recorded for iterations 32–60. Checkpoint 31 passed an exact-source 173-test rerun at final packaging because its initial log is incomplete, as recorded below. Source tags preserve all 30 checkpoints and the rejected cyclic scheduler. Final 0.16 has explicit 1..8 native budgets, worker-local Rust batch APIs, useful parallel QP outputs, and an opt-in native model path for costly propagation. The default remains one thread.

The useful configuration depends on work size. For the existing small NLPs, eight independent solves provide the strongest throughput gains. The new nonlinear propagation surrogate benefits from parallel work within one solve. Larger standalone QPs benefit from independent-output kernels. Small-pilot BFGS and an unsuccessful inverse schedule were restored to their simpler measured paths. Individual wall-time observations are not monotonically faster.

## Final 1 / 8 results

Eight CPU-equivalents quota, nine visible CPUs, children pinned to eight allowed CPUs on one shared AMD EPYC 9V74 host. Timed child processes run serially; a child can use up to eight numerical workers. No compilation, installation or numerical stress ran concurrently with timing. Ratios are one-worker/eight-worker time. All solves passed unchanged original-unit KKT and known reference checks.

| Workload and timing scope | One worker ms | Eight workers ms | Speedup |
|---|---:|---:|---:|
| Propagation 32, active-set, native solve | 37.18 | 9.12 | 4.078× |
| Propagation 32, active-set, setup-inclusive | 44.44 | 16.19 | 2.746× |
| Propagation 32, clarabel, native solve | 39.40 | 12.20 | 3.231× |
| Propagation 32, clarabel, setup-inclusive | 46.36 | 19.79 | 2.343× |
| Standalone QP 512, three fresh-H calls | 46.45 | 29.33 | 1.584× |
| Standalone QP 1024, three fresh-H calls | 349.11 | 201.00 | 1.737× |
| Orbit20, active-set,16 independent solves | 486.99 | 108.01 | 4.509× |
| Orbit20, clarabel,16 independent solves | 3695.51 | 595.00 | 6.211× |

Propagation figures use seven measured repetitions after one warmup. Workflow includes finer-target model construction, worker setup/teardown and the solve; native solve time excludes model construction and includes its own worker setup. Both modes take 10 SQP iterations and reach the same objective. The three-repeat complete matrix also remains available (active-set native 37.03→10.29 ms versus 37.18→9.12 ms in the seven-repeat confirmation). These are separate campaigns; no slower campaign was discarded.

The propagation fixture integrates independent controlled nonlinear ODEs with RK4 and forward sensitivities. Its arithmetic contributes to the objective/gradient, and a finer-step target defines known controls. Tests check finite-difference sensitivities, integration refinement and bitwise serial/eight-thread outputs. It is a separate surrogate; the unavailable user low-thrust application was not tested. Existing orbit callbacks and mathematical inputs are unchanged.

QP figures use seven repetitions, three changed-Hessian solves per sample with retained resources, so exact factor/inverse cache hits cannot hide the work. Cold resource setup and full process/CPU time are separate raw fields. The harness explicitly owns a resource keeper; tiny fallback QPs need no such pool in the actual API. The 512/1024 cases exercise **standalone QPs**: the NLP cap remains 256. Cholesky and ordered working-set decisions remain serial, limiting speedup.

Batch figures use three repetitions and fixed total 16 solves, with inner 1 and outer 1 or 8. They measure aggregate throughput, not lower latency for each concurrent solve. The complete matrix includes pure outer 1/2/4/8 scaling and mixed outer/inner budgets whose product never exceeds 8. Mixed-cost 32 orbit20/40 solves improve from 1829.37 to 431.81 ms (4.236×) in the final three-repeat campaign. No eightfold-scaling promise is made.

The exported figure in `assets/scaling.png` /SVG uses one consistent three-repeat NLP matrix and the seven-repeat QP matrix. It separates native latency, standalone QP scaling and batch throughput. TABLES.md and data/final-scaling.csv contain every final comparison, including slower observations.

## Improvement relative to the already parallel 0.15 manual baseline

0.15 already allowed manual independent solves. Its immutable checkpoint 31 harness uses independent OS workers and fresh backends per job. The 0.16 harness/API retains one backend per worker and dynamically assigns jobs while preserving input-order results. Models, starts and checks are the same. This internalizes tested orchestration/resource reuse; the gains are not a claim that 0.15 could not be manually orchestrated similarly.

A targeted seven-repeat campaign checks one throughput gain and two potentially slower observations from the full three-repeat comparison:

| Family / size / backend / inner / outer / jobs | 0.15 ms | 0.16 ms | 0.15/0.16 |
|---|---:|---:|---:|
| separable / 128 / active-set / 1 / 8 / 16 | 64.33 | 55.10 | 1.167× |
| orbit / 40 / active-set / 1 / 1 / 1 | 255.30 | 261.07 | 0.978× |
| orbit / 20 / clarabel / 1 / 8 / 16 | 620.39 | 603.58 | 1.028× |

For the separable batch, aggregate CPU time in that campaign is 278.52→300.77 ms while wall time falls 64.33→55.10 ms: higher throughput does not imply fewer CPU-seconds. The earlier three-repeat same-configuration campaign measured87.04→62.54 ms for that batch; both campaigns are retained. Orbit40 single-solve time remains about 2.3% slower in the seven-repeat comparison. The earlier matrix includes active-set single separable 17.29→18.18 ms and Clarabel eight-worker orbit 644.29→679.13 ms; the latter reverses to 620.39→603.58 ms in confirmation. These observations do not isolate code-layout/cache effects from shared-host variability. No uniform existing-fixture latency win is claimed.

## Held-out checks

Old frozen holdouts are orbit30/48, separable96/192 and restoration48/96, with both backends and 1/8 budgets. All measured runs pass the same convergence/reference/KKT assertions; small old fixtures mostly select scalar fallbacks, so their timings remain mixed. New propagation sizes 48/96 were declared before tuning:

| Held-out workflow | One worker ms | Eight workers ms | Speedup |
|---|---:|---:|---:|
| propagation48, active-set | 67.61 | 22.09 | 3.061× |
| propagation48, clarabel | 72.45 | 28.42 | 2.549× |
| propagation96, active-set | 131.94 | 44.19 | 2.986× |
| propagation96, clarabel | 159.50 | 70.62 | 2.258× |

Standalone QP holdouts 384/640/896 also pass their independent KKT and planted primal/dual checks at 1/2/4/8. At640 variables the three-call numeric time is 85.18→55.35 ms; at 896,230.22→138.99 ms. All raw samples, CPU time, setup time and binary hashes are retained. Results apply to these controlled fixtures, not industrial robustness or the actual user trajectory.

## Retained implementation

- Explicit bounded, shared owned execution pools, weakly indexed by width; no mutation of the global Rayon pool, no global numerical cache and no new unsafe code. Small work avoids idle worker creation.
- Maximum native budget through Options/QP capability hooks, plus `NlpProblem::set_threads` for explicit Rust propagation work. Model and observer entry points remain on their caller thread; no Send/Sync requirement is added to model instances. Existing Python model bodies remain serial; native phases release the interpreter as before.
- Parallel independent inverse columns, reflector targets, dense transformed normals/KKT projections, coordinate-tiled primal reconstruction, dense lower-Hessian loading and sufficiently large bounded-row norms. Original within-output scalar arithmetic/order and exact-zero rules remain.
- Rust `batch::solve_many` / `try_solve_many`: worker-local factory construction/backend reuse, dynamic bounded claims, linear indexed collection, per-start factory/input/evaluation errors and shared cooperative cancellation. Factories are Send+Sync; model/backend instances need not be. Budget product is at most 8; completed results/errors remain ordered and later cancelled tasks skip factories.
- Fail-closed timing harness with full finite/reference/residual gates, fixed total work, one warmup, repeated/rotated paired builds, pinned child affinity, separate workflow/native/CPU timing, preserved failures, complete journals and source/extension provenance. Worker setup now counts toward the native solve deadline and is skipped on pre-cancel/expiry.

Exact bit-key factors/inverses, original-model validation and independent acceptance from 0.15 remain mandatory. Changing H invalidates the caches; rows/bounds/gradients receive fresh processing. Scalar BFGS is byte-identical to 0.15. See VALIDITY.md for the detailed arguments and scope, REPRODUCE.md/docs/api.md for usable APIs and commands, and each linked plan/result for its own changes/evidence.

## Validation of the delivered numerical code

| Check | Outcome |
|---|---|
| Regular Rust workspace | 196 tests pass in debug/release, plus explicit propagation example tests; iterations 32–60 ran full release workspace tests before timing; exact checkpoint 31 passed all 173 tests at final packaging because its initial log is incomplete. |
| Forced all-target/all-feature Rust | 211 tests pass, including kernel/propagation examples; relocated libpython is handled by a private linker path. |
| Compiled Python | 156 tests pass, including 1/2/4/8 widths, caller-thread ownership, exception identity, cancellation, independent callers and deliberate false-performance fault injection. |
| Clippy / rustfmt / rustdoc |All-target/all-feature warnings-denied Clippy, formatting and warnings-denied documentation pass. |
| Full three-way numerical audit |52,116 actual calls:17,372 each for immutable0.15 baseline1,0.16 scalar1 and 0.16 budget 8;34,744 paired comparisons. No lost convergence/acceptance/reference, no false convergence claims, no changed outer iteration counts; maximum audited accepted objective and point differences both 0.0. |
| Convergence counts |16,638 converged per mode;734 existing nonconverged outcomes per mode remain visible. Call totals include repeated configurations, not unique mathematical problems. |
| Additional development audit |2,760 calls, with 895 of 920 converged per mode, identical outcomes/accepted points/objectives; raw quick audit retained separately. |
| Harder QP census |1000 cases, seed 2026, baseline/scalar/eight-thread budget outcome tallies match, including four active-set independent-check rejections and one limit. These are not successful solves. Dimensions≤100 use small-kernel fallbacks; separate 512+ exact/stress/reference tests exercise actual parallel kernels. |
| Integrity / isolation |Eight frozen callback/generator/evaluator files match 0.15. Audit checks exact case/mode completeness, observed call counts, versions, consistent native extension hashes and actual solver-owned multiplier certificates. Concurrent callers change H/rows/bounds/gradients, inject invalid input/cancellation and recover on later solves. |

The full corpus covers six 300-model seeds under three settings with both backends, an elastic seed, default/SOC/augmented/Jacobian-scaling variants over baseline/Maratos/linear/MPCC families, and independent NumPy checks of development/heldout fixtures. Every observed solve is journaled; variant objectives/points are now audited as well. Convergence is numerical first-order acceptance, not global optimality or a second-order certificate.

## Experiment and failure ledger

1. Parallel inverse work first lost at 128/256 but helped 512/1024. Thresholds were raised; one 256-variable comparison recovered 546.50→115.80 ms after avoiding fine-grained QP dispatch. That was a correction of an intermediate candidate, not an improvement over 0.15.
2. Parallel BFGS triangle/commit prototypes passed bitwise tests but lost within the 256-variable NLP pilot. Iteration 58 restores the full 0.15 scalar implementation. Seven-repeat eight-thread budget separable256 time 118.26→111.74 ms and CPU 135.32→113.86 ms support that removal; scalar samples remained noisy. Prototype tags and all slower samples remain.
3. A cyclic inverse-column schedule gave no reliable retained gain. Two seven-repeat 1024 comparisons showed roughly 5–10% losses; a hybrid revision remained noisy. Iteration 58 removes the extra lane allocation and keeps the simpler generic schedule. `sqp-parallel-52-cyclic-experiment` and all initial/recheck/retained measurements are included.
4. Harness/test integration failures were corrected and fully rerun: an unnecessary unwrap of a scalar objective, a nonexistent CSR helper in a new test, a Clippy temporary-vector warning, and an incomplete first removal of the BFGS driver calls. First-failure logs are retained; none is counted as a passing checkpoint.
5. The 0.15→0.16 comparisons include flat/slower cases and confirmations; no selectively replaced campaign or multiplied adjacent speedups is used. The one-measured-repeat iteration 56 matrix is exploratory; final matrix/holdout/common comparisons use three repetitions, selected confirmations/QPs seven.
6. The initial checkpoint 31 Rust log ends during compilation, so it cannot certify that run. Its mathematical implementation is unchanged0.15 and all initial benchmark assertions passed; the final package audit reran the exact saved checkpoint 31 full release suite (173 passed) and retains both logs. Original logs for 35/49/57 contain all regular test results but end during zero-test rustdoc phases; final-release rustdoc and complete final suites pass separately. The ledger marks this retrospective verification explicitly.

## Delivery and practical next use

Use eight outer workers with inner 1 for the existing small NLPs. Use the native model budget hook when expensive independent propagation dominates. Explicit threads 8 is a maximum; cheap existing Python callbacks do not become parallel automatically. Clarabel's engine and full BFGS remain serial. The 256-variable NLP cap, dense Hessian and Linux-only executed validation remain material limits. Rust 1.85 minimum and other platforms were not separately run; executed toolchain is Rust 1.99.1/CPython 3.12.14.

The archive includes complete source/tests, both the incremental0.15→0.16 and cumulative original-upstream→0.16 patches, source-history bundle, all 30 plans/results, raw measurements, full/quick numerical journals, successful/failed validation logs, figure/CSV and byte checksums. Source-only delivery avoids machine-specific executables; REPRODUCE.md documents rebuilding. The previous0.15 artifacts remain intact, and no upstream push/release publication was performed.

Primary implementation references: [Rayon 1.12 local ThreadPoolBuilder](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPoolBuilder.html), [ThreadPool execution](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPool.html), and the [original SQP-craft repository](https://github.com/dietmarwo/SQP-craft). Numerical safety arguments and actual claims above are grounded in the delivered code, independent checks and raw measurements.
