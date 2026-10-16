# SQP — iteration 1 implementation plan

> Archived pre-implementation plan. The implementation and completed checks
> are recorded in [iteration-1-results.md](iteration-1-results.md).

Planning2027 · Idea 004 · Revision 2 · 16 October 2026 (UTC)

**Goal:** implement a small native Rust SQP driver behind the checked-in nonlinear interface, preserving grouped callbacks, signed multipliers, QP outcomes and observer/error semantics. Use a replaceable convex QP backend, damped BFGS and an L1 merit line search. Establish correct behavior on declared small problems before pursuing trajectory-scale performance.

**Verified source:** public `dietmarwo/SQP-craft`, branch `main`, commit [`d2d2cf117edd9d3c78a1e78416295ec500f2bb31`](https://github.com/dietmarwo/SQP-craft/tree/d2d2cf117edd9d3c78a1e78416295ec500f2bb31), “Add SQP interface draft: Rust contracts, PyO3 binding, dummy backend (#1)”. The clone was clean; no repository or applicable ancestor `AGENTS.md` was present. This revision supersedes the archive-based plan and its incompatible callback, multiplier and observer proposals.

## 1. Checked-in state and evidence

| Evidence | Verified state |
| --- | --- |
| Source tree | 33 tracked files; core and binding crates; no real NLP/QP solver |
| Core/binding | `sqp-craft` / `sqp-craft-python` 0.1.0; edition 2024, MSRV 1.85; dependency-free core with `forbid(unsafe_code)` |
| PyO3 | Manifest allows `0.30.1`; committed lockfile resolves 0.30.1; `abi3-py310` |
| Python distribution/import | `sqp-craft` / `sqp_craft`; Python >=3.10 |
| Licensing | MIT in `LICENSE` and package metadata; `publish = false` |
| GitHub CI | [Run 38835539400](https://github.com/dietmarwo/SQP-craft/actions/runs/38835539400) reports success for this exact commit |
| Local Rust checks | fmt, Clippy with `-D warnings`, 32 integration tests, 1 doctest and the toy example passed |
| Local Python checks | 42 tests and the Python example passed against a directly compiled release extension |
| Local environment | Linux x86_64, Rust 1.99.1, CPython 3.12.14; committed PyO3 lockfile; pytest 9.1.1, NumPy 2.5.4 |

The local check script passed its Rust stages but could not complete maturin packaging: the installed maturin 1.15.0 binary lacked execute permission and then crashed after that installation issue was corrected. The binding was then built with the documented `PYO3_BUILD_EXTENSION_MODULE=1 cargo build --release --locked -p sqp-craft-python` path, staged with the unchanged Python package in a separate directory, and tested there. This tested the actual native module, not a Python substitute. The local wheel path, declared MSRV and other CI platforms were not independently reproduced; remote CI success is separate evidence.

The concept source is `open-rust-solvers-concept-v3.md`. The low-thrust application, actual dimensions, derivatives, reference output and performance target are not in this review. `docs/target-card.md` intentionally remains incomplete.

### Existing interfaces to preserve

| Location | Checked-in behavior | Consequence |
| --- | --- | --- |
| `crates/core/src/spec.rs` | Validated immutable `ProblemSpec`; canonical CSR; finite, box-feasible `check_point` | One spec can serve multiple solves; no repeated structure validation or silent start projection |
| `crates/core/src/problem.rs` | `functions(x, c_out) -> f` and `derivatives(x, g_out, j_out)`; solver-owned buffers | Retain grouped calls; derivatives must follow functions at the same point |
| `crates/core/src/dummy.rs` | One functions call and one derivatives call; NaN-prefilled buffers; candidate only after full evaluation | Preserve accounting, cancellation checkpoints and no false convergence |
| `crates/core/src/solver.rs` | `SqpSolver` trait, `Multipliers`, grouped `EvaluationCounts`, options, observer and results | Add a `NativeSqp` implementation; do not replace the public protocol |
| `crates/core/src/kkt.rs` | Independent KKT checks in the signed convention `grad f = J^T lambda + z` | Reuse and harden, rather than creating a different multiplier convention |
| `crates/core/src/qp.rs` | Lower-CSR Hessian, `QpOutcome`, multipliers, active set, rich `QpWarmStart`, `QpModel::residuals` | Outcomes/duals/working-set types already exist; add capabilities and real implementation |
| Binding and `python/sqp_craft` | Immutable comparable/picklable native spec; copied callback lists; `x0` only; `UndefinedPointError`; observer `True` stops | Preserve the committed semantics, including original fatal Python exceptions |
| CI, tests and `_native.pyi` | Existing checks and platform matrix; derivative and KKT fixtures | Extend, do not recreate or describe as absent |

There is no optimization loop, real QP backend, Hessian update, reusable derivative-checking utility or real application acceptance suite yet. The existing independent KKT evaluator and QP seam are substantial completed work.

### Findings that affect iteration order

1. **KKT overflow can be reported as zero.** A direct probe using a two-row, one-column Jacobian `[1e308, -1e308]`, row multipliers `[1e308, 1e308]`, zero point, gradient 1 and zero equality rows made `J^T lambda` NaN. `kkt::residuals` returned all residuals zero, although the exact stationarity residual for those supplied multipliers is 1. Guard every intermediate arithmetic result before reducing maxima; a finite-input overflow must produce `Error::Evaluation`, never convergence.
2. **QP warm starts validate lengths only.** A step containing NaN and variable multiplier containing infinity passed `QpWarmStart::validate`. Reject nonfinite supplied primal/dual data before any adapter call. Validate active-set meaning separately from shape.
3. **Raw bounds helper remains conditional on prior validation.** `Bounds::new(vec![0.0], vec![]).violation(&[-1.0])` returned zero. Immutable specs protect the normal solve path, but direct public helpers need checked boundaries or explicit/private validated-only use.
4. **Recoverable Python errors need a new lifecycle in the real loop.** Currently `PythonModel::keep` stores every exception and the binding rethrows it after solve. That is correct for a dummy that aborts immediately. A real line search must consume/clear a recovered `UndefinedPointError`; otherwise a later successful solve would still rethrow a stale exception. Fatal exceptions and undefined initial points must keep their original type/traceback.

Items 1–3 were reproduced against unmodified committed cores. Item 4 is a source-based integration requirement, not a claim that the existing dummy violates its documented behavior.

## 2. Iteration scope and gates

Follow the checked-in design notes: probe **HiGHS active-set QP first**, behind the existing `QpBackend`; keep Clarabel as the interior-point comparison or an explicitly documented fallback if the required HiGHS API is unavailable. Implement the outer driver in Rust. Pin and compile-test Hessian input, QP duals, limits, cancellation and available warm-start kinds before committing to an adapter. Do not assume the existence of `ActiveSet` in this repository means the external API can import/export or reuse one. A cold-start adapter is acceptable if its limitations are explicit. This differs from the previous plan’s unconditional Clarabel-first choice and aligns with the repository’s comparison goal.

The proposed pilot supports smooth objectives and constraints with supplied first derivatives, general two-sided rows, equalities and variable bounds. Sparse Jacobian input is retained. The pilot Hessian is dense damped BFGS; this is deliberately not a large sparse SNOPT replacement. Propose a configurable default dense-pilot cap of 256 variables, with a checked memory estimate before allocation. That cap is a development scope choice, not an inferred trajectory size.

Use one backend thread and one model instance per solve. Do not combine a trust-region method, filter, augmented-Lagrangian globalization and an L1 merit line search in the same first iteration. The selected method here is a convex-QP SQP with an L1 merit line search.

### Required first-iteration behavior

- A genuine sequence of accepted SQP steps on the analytic acceptance suite.
- Feasibility, stationarity, dual-sign and complementarity diagnostics at the returned point.
- Correct failure reporting for infeasible linearizations, failed QPs, no descent, stalled line search and resource limits.
- A reusable directional derivative checker for the existing callback contract.
- Correct Rust/Python ownership, error propagation and cooperative cancellation.
- One open nonlinear baseline on the same test models and starts.

### Deferred

Elastic/restoration subproblems, rank-revealing constraint reduction, second-order correction, filter/trust-region globalization, exact Hessian callbacks, limited-memory or block BFGS, automatic scaling, AD, parallel derivatives, factorization reuse and backend-specific active-set persistence beyond verified capability, full trajectory validation and performance parity claims.

Without restoration, the pilot may stop on an infeasible linearized QP even when the nonlinear model is feasible. That is an explicit supported failure outcome. Do not conceal it with random restarts or by calling the nonlinear model infeasible.

### Relationship to concept v3

S0–S3 create useful contracts, derivative tools and a tested QP adapter. Before S4–S7, record whether the real open baseline exposes a relevant gap and which track is primary. If application data are still unavailable, the native loop may proceed only as a bounded analytic research pilot; its result must remain labelled as such. Planning both projects does not commit to simultaneous full solver development.

## 3. Extend the contracts before writing the loop

### 3.1 Explicit solver selection

Proposed Python call:

```python
result = sqp_craft.solve(spec, model, x0, method="sqp", options=options)
```

Add keyword-only `method`, supporting `"dummy"` and `"sqp"`; retain `"dummy"` as the draft compatibility default. Real examples specify `"sqp"`. Return method and QP backend/version metadata. Missing compiled support is an explicit error; there is no fallback to a dummy or a different algorithm. Bump the extended draft to 0.2.0 and update Python type information and examples together.

Both the dummy and real method require finite x0 within variable bounds, exactly as committed; x0 may violate nonlinear rows. Keep `x0` as the sole primal-start argument. Do not resurrect the archive’s redundant NLP `warm_start` parameter. Equal finite variable bounds are allowed and represented as equalities in the QP.

### 3.2 Outer result and termination

Extend status with `Converged`, `IterationLimit`, `EvaluationLimit`, `TimeLimit`, `QpFailure`, `LinearizationInfeasible`, `LineSearchFailure`, `NoDescent`, `Stalled` and `NumericalFailure`, while retaining `Stub` and `Cancelled`. Preserve `Cancelled` for an observer stop and add a structured reason if differentiation is needed; do not introduce an incompatible observer status. Invalid input and fatal callback exceptions remain errors; an undefined trial point is recoverable as specified below.

`Converged` means the declared first-order KKT tolerances are met at the returned x. It does not establish global optimality or exclude a saddle point. A small step, a small objective change, QP success or primal feasibility alone is insufficient.

Results retain the last fully evaluated accepted point and its matching objective/constraints/derivatives. Retain existing `multipliers`, `stationarity`, `complementarity`, and `counts.functions`/`counts.derivatives`. Add dual-sign violation, backend/QP status, termination reason, last accepted step norm, total QP solves/iterations and timing categories. Missing quantities remain `None`; do not fill them with zero. Error propagation preserves the original Python exception and traceback.

### 3.3 Extend the existing QP seam

Keep `QpOutcome::{Optimal(QpSolution), Infeasible, Unbounded, LimitReached(Option<QpSolution>), Cancelled}`. Keep `QpSolution { step, multipliers, active_set, iterations }`, `QpWarmStart { step, multipliers, active_set }`, and `QpModel::residuals`. Do not replace these with the obsolete primal-only shape assumed by the previous plan.

Add explicit per-solve QP options (constructor configuration or a carefully versioned options argument) for tolerance, remaining deadline and iteration cap; the current trait has no options argument. Record backend/version/raw status, the kind of a limit, solve time and warm-start handling in an adjacent report or compatible result extension. If reduced-accuracy/numerical-failure/unknown outcomes need new variants, add them with explicit semantics and update adapters exhaustively. A backend API failure remains an error, not infeasibility.

Accept `Optimal` only after the hardened `QpModel::residuals` passes independent primal, stationarity, complementarity and sign thresholds. Do not use `LimitReached` iterates or reduced-accuracy solutions for steps by default. `Infeasible` maps to `LinearizationInfeasible`, not nonlinear infeasibility. With SPD B and feasible constraints, an unbounded report indicates inconsistent data, a conversion problem or numerical failure; it does not authorize an arbitrary step.

Expose backend capabilities for supported warm-start components, thread behavior and interruption. Validate all supplied data even if the token is already cancelled. Reject unsupported start components, or require an explicit documented ignore policy; never silently claim reuse. Start the outer driver cold. Add primal/dual/active-set reuse only after measured API support and cold-versus-warm equivalence tests; factorization reuse is not implied. A future Clarabel adapter must follow the same policy.

## 4. Mathematical conventions

### 4.1 Preserve the repository multiplier convention

The public representation has one signed multiplier per original row and one per variable:

```text
grad f(x) = J(x)^T lambda + z
L(x, lambda, z) = f(x) - lambda^T c(x) - z^T x
```

Positive multipliers select active lower bounds; negative multipliers select active upper bounds; equality/fixed-variable multipliers may have either sign. Inactive and free entries have zero multipliers. Use this convention in QP results, NLP results, BFGS and tests. The previous plan’s public split nonnegative multipliers and opposite-sign Lagrangian are withdrawn.

Reuse the checked-in definitions: primal is the largest row/variable bound violation; stationarity is `norm_inf(grad f - J^T lambda - z)`; complementarity is the largest `abs(mu) * abs(value - selected_bound)`; dual-sign violation is the largest multiplier magnitude whose sign selects an infinite bound. Harden nonfinite arithmetic first. For equalities the distance to the shared bound supplies the existing residual; preserve this behavior.

Backend-internal split rows are allowed, but require a reversible mapping. If nonnegative conic duals are `y_lower`, `y_upper`, return `lambda = y_lower - y_upper`; for an equality with conic stationarity `g + A^T y = 0`, return `lambda = -y`. Apply the same rule to variable-bound rows. Free rows map to zero. Do not expose an internal split-dual convention as the public result.

All checks use the original units. There is no automatic scaling in this pilot. Keep separate residuals and absolute tolerances; real mixed-unit trajectory acceptance requires the target card and an independent unscaled evaluator.

### 4.2 Build the QP in step coordinates

At x_k, with g_k = grad f(x_k), J_k = dc/dx and SPD B_k:

```text
minimize_p  0.5 * p^T B_k p + g_k^T p
subject to  lower_x - x_k <= p <= upper_x - x_k
            lower_c - c(x_k) <= J_k p <= upper_c - c(x_k)
```

The shifts are essential. Passing original variable bounds or original row bounds directly to a step QP is incorrect. Keep numerical Jacobian zeros in the fixed declared sparsity pattern.

### 4.3 Backend conversion and capability probe

For HiGHS, use the selected API’s documented triangular Hessian and sparse matrix layout, converting from the repository’s lower-CSR representation without doubling off-diagonal entries. Compile a non-diagonal QP probe; do not infer QP support from the LP wrapper alone. Prefer direct two-sided rows, original minimization sense and native row/column duals in the shared convention. Verify this correspondence with hand-derived lower, upper, equality and fixed-variable cases. Detect integer index overflow and finite values that collide with backend infinity sentinels. Confine any necessary C API calls to the adapter crate; preserve the safe core.

For a later Clarabel comparison, convert lower-CSR Hessian entry `(i,j,v)` to upper-CSC `(j,i,v)` without doubling. Its conic rows are `A*p+s=b`: upper side `(a,r)` with nonnegative slack, lower side `(-a,-l)`, equality `(a,r)` with a zero cone. Omit infinite sides. Retain a reverse map and reconstruct signed row/variable multipliers as in section 4.1. Use documented unscaled solution fields, not scaled internal arrays. This comparison is optional in iteration 1, not a second mandatory adapter.

Every adapter is checked against the original two-sided `QpModel`, including objective, shifted bounds, off-diagonal Hessian products and mapped KKT residuals. The backend’s internal success flag alone is insufficient.

## 5. Native SQP algorithm for this iteration

### 5.1 State and initialization

One solve owns its iterate, last complete evaluation, B, penalty parameter, multipliers, counters and QP workspace. Validate the fixed sparsity pattern once, then validate callback value dimensions/finiteness on every evaluation. Apply the numerical-boundary and raw-helper fixes from section 1 before trusting convergence. `ProblemSpec` already validates structure once; the driver validates only per-call values and new QP data.

Start B at a positive diagonal multiple of the identity, default 1 in normalized pilot units. Use existing linear-algebra code for SPD checking if needed; do not implement a new factorization library. Estimate O(n²) allocations and enforce the configured dense-pilot limit. Start multipliers at zero and penalty rho at a documented positive default.

### 5.2 Outer iteration order

1. Check cancellation, deadline and evaluation/iteration budgets.
2. Obtain a complete current evaluation; reuse cached values when x has not changed.
3. Build and solve the convex step QP; verify the step and multipliers.
4. Recompute current-point KKT diagnostics using current derivatives and the fresh QP multiplier estimate. Return `Converged` only if primal, stationarity, dual-sign and complementarity tests all pass.
5. Compute a merit decrease prediction and choose/update the penalty as below.
6. Backtrack along p, calling grouped `functions` for each trial and handling only explicit `UndefinedPoint` as recoverable.
7. At a merit-acceptable trial, call grouped `derivatives` immediately after its successful `functions` call. Promote only after derivatives are valid; if explicitly undefined, backtrack from that trial instead.
8. Update B using the accepted displacement and a consistent Lagrangian-gradient difference.
9. Emit one accepted-iteration observer event and repeat.

Prefill every reused output buffer with NaN, as the dummy does. Do not perform derivative callbacks at merit-rejected trial points. Every derivatives call must use the x of the immediately preceding functions call, including derivative checks, retries and cache restoration; a cached value alone does not establish that model-side precondition. Do not report an unevaluated trial as the final solution. If a budget expires while completing a trial's derivative data, return the preceding fully evaluated accepted point and preserve the consumed counts.

### 5.3 L1 merit and line search

Let v(x) be the sum of absolute equality residuals and positive inequality residuals for nonlinear rows. Variable bounds are maintained from a bound-feasible start and a checked step; still include them in final feasibility. Numerical QP feasibility alone is not enough to uphold the strict callback box contract. Before every callback, check the computed trial with `ProblemSpec::check_point`. Restrict alpha to the representably feasible interval; if a step points out of an already active bound, tighten/re-solve the QP or fail explicitly rather than clipping x or evaluating outside the box. Fixed variables must remain fixed. Recompute the merit prediction for any deliberately changed step. Test this with a callback that asserts its domain on every call.

```text
phi_rho(x) = f(x) + rho * v(x)
v_lin(x,p) = violation sum of linearized nonlinear rows at full step p
predicted_decrease = -grad_f(x)^T p + rho * (v(x) - v_lin(x,p))
```

Use the maximum absolute signed constraint multiplier to propose a nondecreasing penalty, for example `rho = max(rho, 1.1*norm_inf(lambda) + 1)` (zero for no nonlinear rows). This is a pilot policy to validate, not a claimed convergence theorem. Check predicted decrease explicitly. If it is nonpositive beyond roundoff, permit one documented positive-diagonal Hessian reset and QP retry; then return `NoDescent` if the issue persists. Account for the retry.

Start alpha=1 and halve it. Accept when:

```text
phi_rho(x + alpha*p) <= phi_rho(x) - eta * alpha * predicted_decrease
```

Proposed eta is `1e-4`, with at most 25 backtracks. Keep rho fixed during a single line search. A failed line search returns `LineSearchFailure`; a tiny step without acceptable KKT residuals is `Stalled`. The Maratos effect and missing second-order correction are known pilot limitations, not reasons to weaken feasibility tolerances.

Use the existing `UndefinedPoint` / `UndefinedPointError`: an undefined initial point aborts; an undefined trial functions or derivatives call consumes its count, invalidates trial buffers/model cache and triggers shorter-step retry within the same budget. Do not call derivatives after failed functions. If all trials are undefined, report `LineSearchFailure` with the last complete accepted point and a specific reason. All other exceptions and nonfinite/malformed outputs abort with their original error semantics. A recovered Python domain exception must be cleared from binding state; a subsequent fatal exception must be the one rethrown.

### 5.4 Damped BFGS

For accepted displacement s, use the same new multiplier estimate at both endpoints:

```text
y = [grad_f(x_new) - J(x_new)^T lambda_new - z_new]
  - [grad_f(x_old) - J(x_old)^T lambda_new - z_new]
t = s^T B s
theta = 1                         if s^T y >= 0.2*t
theta = 0.8*t / (t - s^T y)       otherwise
r = theta*y + (1-theta)*B*s
B_new = B - (B*s)*(B*s)^T/t + r*r^T/(s^T r)
```

The constant variable multiplier contribution cancels. Use the accepted QP multiplier estimate consistently for this update; do not mix old multipliers at one endpoint with new multipliers at the other. The next QP refreshes the estimate before the next KKT decision.

Guard tiny/nonpositive denominators and nonfinite arithmetic. For negligible accepted displacement, skip the update with a recorded reason. Symmetrize against roundoff and verify SPD; on failure perform a bounded positive-diagonal reset and count it. Repeated failures return `NumericalFailure` rather than growing an arbitrary regularization forever. No indefinite Hessian is sent to the convex backend.

## 6. Limits, callbacks and Python execution

Proposed pilot options: `max_iterations=100`, `max_backtracks=25`, `max_callback_calls=2000`, optional wall-time limit, separate feasibility/stationarity/complementarity/dual-sign tolerances, QP tolerances/iteration cap, and a dense-size limit. These are experiment settings, not trajectory requirements. Every option must be used or rejected; keep the dummy’s supported options distinct from real-method settings.

Count `functions` and `derivatives` invocations separately, preserving `EvaluationCounts`. The total callback budget is their sum, including undefined/rejected/discarded trials and cache-restoration calls. Check before every invocation and increment for attempted calls. Count QP retries/backend iterations separately from accepted SQP steps. Use a monotonic deadline and pass remaining time into the QP adapter. An in-flight Python callback cannot be forcibly interrupted safely. If limits occur during derivative completion, return the preceding complete iterate, never partial trial buffers.

The current binding stores borrowed `Bound<'py, PyAny>` callbacks and keeps the interpreter attached for the whole dummy solve. For long native work, convert to owned `Py<PyAny>` references, clone the cancellation token and construct worker-local model/observer adapters inside a detached closure. Reattach for grouped callbacks, signals and output conversion. Borrow immutable Rust spec data safely or use an `Arc`; do not move borrowed Python tokens into detached work. Keep callback order and same-thread model semantics. Verify this with committed PyO3 0.30.1; neither `unsafe Send` implementations nor a global `Send + Sync` requirement on `NlpProblem` are warranted.

Separate recoverable domain errors from fatal stored `PyErr` values. At the initial point rethrow the original `UndefinedPointError`; at a recoverable trial consume it and clear the pending Python exception, retaining only bounded diagnostic information. Fatal model/observer exceptions retain original identity/type/traceback. Add a test where an undefined trial is followed by successful convergence, and another where a later fatal exception follows a recovered domain error. The existing unconditional post-solve `model.error` rethrow must not survive unchanged into a recovering loop.

Python callbacks still receive independent list copies. Keep the native immutable/picklable `ProblemSpec`, two-method protocol and existing `_native.pyi`; update fields together. NumPy zero-copy and batch evaluation remain measurement-led later work.

Observer semantics are already fixed: `True` stops; `None` or `False` continues; other return types fail. Preserve `observer=print`. Stop returns `Cancelled` with the last complete candidate. Emit initial iteration 0 and one event per accepted step. QP interruption callbacks poll only Rust cancellation/deadline state; do not invoke the Python observer at arbitrary backend checkpoints. No sign inversion from the old archive observer rule is needed.

If a safe detach/reattach design cannot be delivered within the time box, label the binding a serial attached pilot and record its cancellation/throughput limits. Process workers remain the initial parallel execution policy; no GIL-free callback performance claim follows from detaching native phases.

## 7. File-level work packages

| ID | Concrete changes | Acceptance gate |
| --- | --- | --- |
| S0 — baseline handoff | Inspection is complete here; confirm HEAD at implementation start, record local/CI checks and packaging limitation; preserve `docs/target-card.md` knowns/unknowns | Starting commit and actual baseline log, no repeated archive reconstruction |
| S1 — hardening/contracts | Fix arithmetic in `kkt.rs`, `QpWarmStart::validate` in `qp.rs`, and raw helpers in `spec.rs`; extend `solver.rs`/`error.rs` and facade/stub; retain signed multipliers and QP outcomes | Overflow cannot masquerade as zero; nonfinite starts rejected; existing 32 Rust tests plus doctest and Python suite remain green |
| S2 — derivative tools | Add `crates/core/src/derivative_check.rs` and Python entry point; expose existing `kkt::nlp_residuals` in Python; grouped directional checks with cache ordering and evaluation accounting | Correct derivatives pass; deliberately wrong signs, scales and structural omissions fail |
| S3 — QP probe/adapter | Probe HiGHS first; add proposed `crates/highs-qp/{Cargo.toml,src/lib.rs,src/convert.rs,src/result.rs}` plus narrow FFI only if needed; pin dependencies, options and capability report | Non-diagonal QPs, signed duals, outcomes, limits and cancellation pass; supported start kinds demonstrated; any alternate adapter choice recorded |
| S4 — minimal loop | Add `crates/core/src/sqp.rs`, `evaluation.rs`, `merit.rs`; initially use positive identity B | Real accepted steps on simple convex/equality fixtures; budgets and final-point consistency pass |
| S5 — Hessian update | Add `bfgs.rs`, SPD/reset policy and derivative-cache tests | Damped update invariants and nonquadratic progression; no regression from S4 |
| S6 — Python integration | Replace hard-coded dummy dispatch for explicit method; owned grouped callback adapters, recoverable exception lifecycle and GIL boundaries; update facade/stubs | Installed native wheel solves fixtures; callback exceptions, signals, cancellation and mutation isolation pass |
| S7 — evaluation and handoff | Add baseline comparison, per-solve logs, existing CI extensions, real example and updated `docs/api.md`/`docs/design-notes.md` | Complete analytic acceptance suite and honest application status |

Order: S0 → S1 → S2/S3 → S4 → S5 → S6 → S7. S2 and S3 are logically independent after S1, but all integration must use the same multiplier conventions. Do not proceed to BFGS until the constant-SPD-Hessian loop and QP conversion are understandable and tested.

Proposed time box: 10–12 focused working days, with approximately 1 for S0/S1, 1–2 for S2, 2 for S3, 2–3 for S4/S5, 1–2 for S6 and 2 for S7. If this is too large for the current cycle, end the first delivery after S3 with a real QP backend and derivative tools, explicitly marked as not yet an SQP solver. Do not call that reduced deliverable complete outer-interface implementation. Twice the allocated budget for any stage triggers a written diagnosis/re-scope decision.

## 8. Numerical acceptance suite

### 8.1 QP tests before SQP

- SPD quadratic with known unconstrained minimizer; compare to a direct dense solution in the test oracle.
- Equality, lower-active and upper-active cases with hand-derived multiplier signs.
- Two-sided rows and fixed variables; reconstruct original-row multipliers.
- Symmetric off-diagonal Hessian example to detect triangle/transposition/doubling errors.
- Empty Jacobian, redundant rows, impossible equalities, finite-bound shifts and negative coefficients.
- Reduced-accuracy, failed and interrupted results; verify no unjustified step is used.
- Cancellation inside a running QP and time-limit accounting; deterministic status-mapping fixtures supplement timing-sensitive integration checks.

### 8.2 NLP fixtures

| Problem | Reference or required behavior |
| --- | --- |
| Minimize `(x-1)^2 + (y-2)^2`, no rows | `(1,2)`, objective 0, small stationarity |
| Same objective with `x+y=1` | `(0,1)`, objective 2; equality multiplier -2 under the committed convention |
| Minimize `(x+1)^2` with `x>=0` | x=0, objective 1; lower-bound multiplier +2 |
| Minimize `0.5*((x-2)^2+y^2)` subject to `x^2+y^2<=1` | `(1,0)`, objective 0.5; signed upper-row multiplier -0.5 |
| Feasible point with nonzero stationarity | Must take steps or report a failure/limit, never converge on feasibility alone |
| Equality `x^2=1` from x=0 | Linearization is impossible although nonlinear feasible points exist; report `LinearizationInfeasible`, not nonlinear infeasibility |
| Nearly dependent rows and deliberately poor scaling | Finite diagnostics and transparent failure if necessary; no silently relaxed tolerances |
| Wrong gradient sign, permuted Jacobian values, missing structural derivative | Derivative checker detects errors; solver never repairs them silently |
| Undefined trial followed by a valid smaller trial | Backtrack, consume the domain error, and continue without a stale Python exception |
| NaN/Inf/fatal exception in a model or observer callback | Abort with documented error semantics and original Python exception |
| Strict box-domain model and active/fixed bounds | Every functions/derivatives call remains inside bounds, including floating-point step errors |
| Overflowing finite-input KKT products and nonfinite QP starts | Checked numerical/input errors, never zero residual acceptance |
| Limits during line search or derivative completion | Return a fully evaluated accepted point with correct consumed counts |

For well-scaled analytic fixtures, propose primal violation <=1e-7, stationarity <=1e-6, complementarity <=1e-6 and dual-sign violation <=1e-8, with objective error <=1e-6 times `max(1, abs(reference))`. Configure the QP more tightly where needed and freeze the thresholds before tuning. Use several declared starts and require at least one fixture to take multiple accepted steps. Do not loosen thresholds to disguise an integration failure.

An independent Python evaluator recomputes f, c and KKT residuals at returned points and mapped multipliers; it must not call the existing Rust KKT helper. A different open nonlinear solver supplies comparative results on exactly the same models and starts. Its own success flag is supplementary evidence, not the acceptance oracle.

### 8.3 Derivative checker design

Use a deterministic set of directions and central finite differences on small fixtures, over a short sequence of relative step sizes. First call `functions(x)` then `derivatives(x)` at the same base point and retain those derivatives; perturbed points use `functions` only. Before any later derivatives request, repeat `functions` at that exact point so a stateful propagation cache is valid. Compare directional `grad_f dot d` and `J*d` against callback differences with absolute and relative tolerances. Do not dense-expand a large Jacobian merely to check a direction. Detect structural omissions by comparing every constraint output, not just declared nonzeros.

Keep perturbations within variable bounds. Use one-sided differences where a feasible perturbation exists; fixed coordinates cannot be checked by feasible perturbations and must be explicitly reported as skipped. Reject zero feasible directions and report inconclusive checks instead of treating them as passes. Account for all checker callbacks separately from optimizer evaluations.

## 9. Parallelism, benchmarks and real application

The pilot uses a worker-local model, Hessian and QP workspace with no shared solver state. For Python dynamics, start with process-per-worker independent solves; send compact starts/results rather than sharing every derivative evaluation. Cap numerical-library threads in each worker. A two-worker smoke test checks isolation; actual 16×1, 8×2 and 4×4 comparisons require the intended 16-physical-core machine and a real workload. No speedup is assumed.

Record end-to-end time, callback execution/attachment-and-conversion time, QP time, BFGS time, counts and failures. Do not label an inclusive callback-boundary timer as a pure GIL-wait measurement; separating wait requires dedicated instrumentation. Keep original Python dynamics and derivatives unchanged for the first comparison. Model porting, batching and transcription changes are separate experiments.

Before trajectory acceptance, complete `docs/target-card.md`: source/data revision, transcription, variable/constraint units and scaling, analytic derivative source, starts, objective/feasibility targets, budgets, reference outputs, continuous-trajectory revalidation requirements and held-out partitions. Measure an available open baseline on the unchanged model. A reduced real case may enter only within the declared dense scope and with fixed independently checked tolerances.

Save representative QP sequences with B, gradient, shifted bounds, Jacobian, dual mapping and tolerances for a later Clarabel-versus-active-set comparison. Such logging should be opt-in and bounded. The committed warm-start interface already carries primal, dual and active-set information; that representation alone establishes neither backend acceptance nor factorization reuse or SNOPT-like efficiency.

## 10. Definition of done

- [x] Actual repository identity and inspected commit reconciled with the archive.
- [ ] Numerical hardening findings fixed before convergence uses KKT checks.
- [ ] `NativeSqp` implements the existing `SqpSolver` trait with the grouped model/derivative protocol and strict box guarantee.
- [ ] QP status, sparse conversion, row shifts, multipliers and independent KKT checks pass.
- [ ] Native loop takes real steps; merit line search and SPD BFGS invariants are tested.
- [ ] Success requires all declared first-order residuals at the returned point.
- [ ] Infeasible linearization and failed line search never become false nonlinear infeasibility/optimality claims.
- [ ] Callback limits, deadlines, cancellation and observer semantics are enforced and counted.
- [ ] Rust and installed Python-wheel cases agree; recoverable undefined trials clear correctly and fatal exceptions retain their original traceback.
- [ ] Derivative checker detects injected errors and reports inconclusive directions honestly.
- [ ] Comparison uses unchanged models/starts and lists every failure and compute cost.
- [ ] Documentation labels the dense pilot and missing restoration/scaling clearly.
- [ ] `docs/iteration-1-results.md` distinguishes analytic acceptance from pending trajectory acceptance and proposes the next measured task.

This delivery is a plan, not a source-code modification. Exact dependency/toolchain pins and empirical results belong in the implementation record once tested.

## 11. References and companion

- Supplied `open-rust-solvers-concept-v3.md`: staged native SQP pilot, convex-QP requirement, warm-start comparison, Python model preservation and stop rules.
- [Verified source tree](https://github.com/dietmarwo/SQP-craft/tree/d2d2cf117edd9d3c78a1e78416295ec500f2bb31), especially `docs/api.md`, `docs/design-notes.md`, `problem.rs`, `solver.rs`, `qp.rs`, `kkt.rs`, bindings and tests: current contracts and gaps.
- Original `rust-sqp-interface-draft.zip` is historical only; four callbacks, observer `True=continue`, absent multipliers and primal-only QP starts no longer describe the repository.
- [HiGHS C API](https://ergo-code.github.io/HiGHS/dev/interfaces/c_api/) and [Rust binding](https://docs.rs/highs/): QP capability-probe starting points; verify against the actual pinned versions.
- [Clarabel Rust model format](https://clarabel.org/stable/rust/getting_started_rs/): CSC/upper-triangle input, conic signs and returned solution fields.
- [Clarabel Rust termination example](https://clarabel.org/stable/examples/rs/example_callback/): callback stop convention and registration.
- [PyO3 documentation](https://pyo3.rs/): inspect documentation and source matching committed 0.30.1 for the proposed detach/reattach boundary; archive 0.27.2 guidance is not the build target.
- LP/IP iteration-1 plan (included in the companion LP-IP-craft iteration-1 ZIP): independent baseline path and shared validation discipline.

The numerical policies, thresholds, work packages and time boxes are proposed design decisions. Their behavior must be established by the acceptance tests; no convergence or performance theorem is claimed for the eventual implementation.
