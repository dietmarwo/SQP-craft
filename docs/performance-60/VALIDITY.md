# Numerical and execution boundaries of 0.16

The mathematical SQP/QP contracts, defaults, tolerances and budget values remain those of 0.15; worker-setup time accounting is corrected as described below. The default native budget is one. This remains a numerical first-order pilot: at most 256 NLP variables, positive-definite QP Hessians, no new global/nonlinear-infeasibility/second-order certificate. Larger 512/1024 tests exercise the **standalone QP seam**, not a larger NLP limit. Full BFGS is byte-identical to 0.15 after measured parallel prototypes were removed. Clarabel's numerical engine stays serial.

## Ownership and worker budgets

`Options.threads` accepts 1..8 as a maximum. The backend capability hook explicitly rejects unsupported multi-thread requests for custom backends. Existing models implement no additional methods: `NlpProblem::set_threads` defaults to a no-op. An opt-in Rust model can partition its numeric outputs with `Execution` while the callback entry point, mutable model and observer stay on their caller thread. The model trait still requires neither Send nor Sync. Python callbacks are not moved to native workers.

`Execution` owns an explicit Rayon pool. Compatible live owners share an Arc through a short, process-local weak registry keyed by width. The registry holds no numerical data and is not the global Rayon pool. The last owner permits pool teardown. A separate budget may own a separate pool; independent callers must manage their combined CPU budget. Large work uses disjoint mutable slices and immutable inputs, synchronously joined before consumers read outputs. No unsafe code was added. Small work avoids unused pool creation.

Batch APIs enforce `outer_workers * inner_threads <= 8`. Use eight outer workers and inner 1 for small NLPs. Private solver state is created on a worker and reused there; models are independently constructed on that worker, so backend/model types need not be Send/Sync. The shared factory must be Send+Sync. `try_solve_many` returns factory errors per input; the convenience `solve_many` wraps an infallible factory. Dynamic bounded claims distribute jobs, then indexed slots collect exactly one result per start in input order. There is no cross-model mutable numerical cache.

Cancellation retains completed ordered results and avoids later factories. Start errors retain precedence. A pre-cancelled batch creates no workers/models. Running callbacks/factories/kernels remain cooperative. Native model worker setup is included in the solve's wall-time budget; pre-cancelled/expired solves skip it. Per-solve deadlines are not a global batch deadline.

## Parallel QP arithmetic

Every QP still validates original input, checks positive definiteness and independently checks accepted steps/multipliers against original data. No tolerance was loosened, nonzero discarded, diagonal approximation introduced or successful solve inferred from a cache hit.

- Inverse-factor columns are independent. Each keeps the original descending triangular solve and four-lane dot arithmetic. Exactly diagonal and small factors retain scalar fast paths. The temporary cyclic schedule was removed after repeated larger-case losses.
- Householder accumulation and full-tail norm remain ordered. Only independent target-column axpy updates are distributed; exact-zero support and sign correction are unchanged.
- Dense transformed normals and KKT projection dots distribute independent output components while retaining each original scalar reduction order. Cheap variable/singleton normals stay scalar.
- KKT primal reconstruction partitions coordinates, then visits columns in the same original order and skips the same exact-zero coefficients. Triangular multiplier solves remain serial; no floating-point cross-worker reduction or atomic accumulation is used.
- Dense lower-Hessian materialization partitions target columns. A validated strictly sorted lower CSR with maximal triangular entry count contains every lower entry, justifying direct indexed reads. Sparse materialization and unused-upper-triangle handling remain unchanged.
- Large bounded-row norms distribute rows using the unchanged scalar norm. Free rows remain zero and do not count toward work estimates.

Exact Hessian keys, conditioning/factor reuse, unrotated-inverse promotion and all miss/failed-factor invalidation rules from 0.15 remain. Changed gradients/Jacobians/bounds receive fresh processing. Thread-budget changes do not change H-dependent mathematical factors. Tests cover bitwise scalar/parallel kernel identity, boundary sizes, invalid data, failed-SPD recovery after promotion, changing rows/bounds/H and concurrent callers with private model/backend state.

## Separate propagation workload

The new native fixture propagates independent controlled nonlinear ODE segments with fixed-step RK4 and simultaneous variational sensitivities. Every integration contributes to the objective/gradient. A finer-step reference defines known target controls outside solver timing; derivatives reuse the immediately preceding propagation. Segment outputs are disjoint and the objective reduction stays in index order. Finite-difference/refinement tests and bitwise serial/eight-thread comparisons cover its sensitivities, propagated values and gradients.

This is a controlled nonlinear integration surrogate, not the unavailable user low-thrust application. The old orbit, separable, restoration and independent random/linear/MPCC generators are unchanged; their frozen hashes are checked. The new workload demonstrates where native model parallelism helps without changing those old problems.

## Evidence and limits

Correctness audits require complete frozen case/mode keys, actual observed solve counts, one consistent extension hash per mode, correct versions and solver-owned multiplier certificates for convergence. Full journaled objectives/points also cover option variants, whose earlier summaries stored fewer fields. Existing nonconverged outcomes and harder-QP rejections/errors/limits remain visible. Repeated configurations are counted as calls, not unique mathematical problems.

Timings distinguish single-solve latency, setup-inclusive workflow, CPU time and fixed-total-work throughput. The server has eight CPU-equivalents quota and nine visible CPUs; timed children are pinned to eight allowed CPUs. It is a shared host, so tiny differences are not reliable wins. No compilation or numerical stress runs concurrently with timed campaigns. No claim of monotone per-checkpoint wall-time gains or eightfold scaling is made. Linux x86_64/Rust 1.99.1/CPython 3.12.14 were executed; other platforms and the declared Rust 1.85 minimum were not separately tested.
