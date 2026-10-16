# Iteration 3 results — QP workspace reuse

Version 0.4.0 · 16 October 2026 UTC. Base: reviewed `main` at
`214f93b0ec0cf407f62fd01ef23df90c7a81fe5c`.
[Plan](iteration-3-plan.md) · [API](api.md) · [Validation](validation.md).

## Repository review and scope

The base includes the iteration 2 review fixes for relative restoration
stationarity, near-bound steps and numerical-noise steps. Its 70 Rust tests,
one doctest and 76 compiled Python tests passed before this work. Those fixes
and all convergence and independent acceptance thresholds are preserved.

Profiling identified QP work as roughly 87–92% of time on the orbit fixtures.
This iteration implements the selected optimization: reuse compatible Clarabel
0.11.1 workspaces and measure the component costs. It changes no dependency
versions and adds no new SQP algorithm. The default remains cold setup.

## Delivered behavior

- `SqpOptions.qp_reuse_workspace` in Rust/Python and `QpOptions.reuse_workspace`
  are opt-in booleans, defaulting to false.
- Each adapter owns at most one workspace. The key includes dimensions,
  P/A CSC structure, cone kinds and the ordered signed dual mapping.
  All P/q/A/b values are replaced on a hit. Structural changes, including
  restoration transitions, rebuild it. Clones have empty caches.
- Existing Clarabel presolve/equilibration defaults remain enabled. A solver
  whose structure changed in presolve is not retained for updates.
- Failed updates discard partial state. A reused solution that is not an
  independently accepted optimum receives at most one fresh retry. Reused
  infeasibility therefore needs cold confirmation before restoration.
  Cancellation and time/iteration limits stop without retry; any retry shares
  the remaining QP budgets. Settings and callbacks are refreshed every time.
- Per-call and aggregate diagnostics report setups, successful updates, cold
  retries, the last retry reason and conversion/setup/update/solve/check time.
  Existing inclusive QP time and backend-call counts keep their meanings;
  iteration counts include all engine attempts.
- Python types, the extension, examples, tests, API documentation and CI cover
  both paths. Versions are consistently 0.4.0.

Reuse preserves allocation, symbolic structure and initial equilibration;
Clarabel still starts primal/dual iterates cold and recomputes numerical
factors. This is not a warm-start interface. Exact cache, retry and report
semantics are documented in [the API](api.md#qp-workspace-reuse).

## Acceptance results

The existing 13-case evaluator requires the reference objective and original
unit KKT residuals. Both cold and reuse modes pass **13/13**, with identical
outer iteration counts in all cases and no cold retries. SciPy SLSQP passes
12/13 in the existing comparison. The infeasible-start HS071 case still
converges in 8 outer steps, including one restoration step; with reuse it
needs 3 setups and 7 updates.

The unchanged random family uses seeds 1–4, 300 problems per seed. Each has a
known feasible point, a random start and nonlinear equality/inequality rows.
The independent test checks first-order residuals; it has no reference global
objective for this family.

| Outcome, over 1,200 problems | Cold | Reuse |
| --- | ---: | ---: |
| Restoration enabled: converged and independently accepted | 1,186 | 1,186 |
| `restoration_failed` | 12 | 12 |
| `restoration_limit` | 2 | 2 |
| Restoration disabled: converged and accepted | 970 | 970 |
| Restoration disabled: `linearization_infeasible` | 230 | 230 |
| Convergence claims rejected by the independent evaluator | 0 | 0 |
| SciPy SLSQP: independently accepted | 1,165 | 1,165 |

Counts match reviewed iteration 2. They do not establish behavior for all
models or claim that every failed trajectory is identical. Full status totals
per seed are in [stress results](iteration-3-stress-results.json); all 13-case
results are in [baseline results](iteration-3-baseline.json) and
[workspace results](iteration-3-workspace-results.json). No acceptance threshold
was relaxed.

## Repeated orbit timings

Installed release abi3 wheel, Linux x86_64, CPython 3.12.14, NumPy 2.5.4,
SciPy 1.18.1, Rust 1.99.1. Five measured runs per mode/case after an acceptance
sweep warms both paths; execution order alternates each repetition. No
compilation or stress process ran concurrently with these measurements.
The cold column uses the same iteration 3 implementation with reuse disabled.

| Metric | 84 variables, cold | 84 variables, reuse | 164 variables, cold | 164 variables, reuse |
| --- | ---: | ---: | ---: | ---: |
| Median wall time (s) | 0.3471 | 0.3127 | 2.5639 | 2.1641 |
| Min–max wall time (s) | 0.3358–0.3676 | 0.2966–0.3275 | 2.3707–2.6026 | 2.1117–2.3411 |
| Outer steps | 104 | 104 | 167 | 167 |
| QP calls | 105 | 105 | 168 | 168 |
| Interior-point iterations | 839 | 870 | 1,446 | 1,444 |
| Setups / updates | 105 / 0 | 1 / 104 | 168 / 0 | 1 / 167 |
| Cold retries | 0 | 0 | 0 | 0 |
| Median conversion time (s) | 0.01276 | 0.01278 | 0.08620 | 0.07671 |
| Median setup time (s) | 0.04129 | 0.00037 | 0.26491 | 0.00158 |
| Median update time (s) | 0 | 0.00131 | 0 | 0.01001 |
| Median engine solve time (s) | 0.23644 | 0.24310 | 1.92016 | 1.83321 |
| Median backend check time (s) | 0.00289 | 0.00295 | 0.01688 | 0.01627 |

Median wall time falls **9.9% and 15.6%** respectively (cold/reuse ratios
1.110 and 1.185). These are local fixture measurements, with visible runtime
variation; they are not general performance guarantees. Phase medians are
computed independently and need not sum to the median total. At 84 variables,
retaining initial equilibration increases inner iterations by 31, even while
setup savings lower total runtime. This supports keeping reuse opt-in.

Reproduce using the installed native package and bench extras:

```bash
python examples/compare_workspace.py --repeats 5 > workspace-results.json
python examples/compare_baseline.py --table
for seed in 1 2 3 4; do
    python examples/stress_random.py --seed "$seed" --count 300 --table
    python examples/stress_random.py --reuse-workspace --seed "$seed" --count 300 --table
done
```

The benchmark fails on a rejected/nonconverged comparison result, never on a
wall-time threshold. The JSON contains every measured run, objective, residual,
call/iteration count and phase time. CI runs one timing repetition as an
acceptance smoke check and exercises both stress modes.

## Review of the delivered iteration

The delivery built and passed its 79 Rust tests, doctest and 85 Python tests.
Its cold and reuse acceptance and stress results reproduce exactly. The review
checked the reuse path against the pinned Clarabel source, then measured where
the remaining time goes. The measurements are on a different, slower 2-core
Linux VM (Rust 1.97.0, CPython 3.13). Phase timers are the evidence below,
because wall time varied by about ±5% between identical runs.

**One defect: reused workspaces carried earlier calls' time into Clarabel's
time limit.** Clarabel compares the sum of its root timers (setup, solve,
post-process) with `time_limit` and resets only the solve timer per call. The
adapter reset the setup timer, but post-processing accumulated over every
reused call. That is about 0.3 ms per QP at 164 variables, 0.05 s over one
orbit run, and it grows without bound in long runs. A reused solve near the
deadline could therefore stop as a time limit with budget left. The adapter
now resets the post-processing timer too. A test runs 300 reused solves and
requires Clarabel's clock never to exceed the call's own measured engine
time; it fails on the delivered code.

**Where the time goes.** Orbit, 164 variables, reuse on, medians of 5 runs:

| Phase | Seconds | Share of wall time |
| --- | ---: | ---: |
| Clarabel engine solve | 2.75 | 83% |
| — KKT numeric factorization (Clarabel timer `kkt update`) | 1.63 | 49% |
| — KKT solves including iterative refinement | 0.73 | 22% |
| — default start (one factorization and solve per QP) | 0.25 | 8% |
| Python model callbacks (binding overhead about 0.02 s) | 0.24 | 7% |
| BFGS update including its Cholesky check | 0.10 | 3% |
| QP conversion including the Cholesky check (delivered: 0.13) | 0.07 | 2% |
| Setup, update, check and driver overhead | 0.09 | 3% |

The factorization takes about 1.1 ms per interior-point iteration. That fits
n³/3 for the dense 164 × 164 BFGS block in QDLDL. Reuse removes the setup
(0.46 s, 12% of a cold run) but cannot reduce this cost.

**Engineering changes, with no numerical effect:**

- Conversion builds Clarabel's CSC data directly in O(nnz). The validated
  lower-triangle CSR Hessian already is the upper-triangle CSC that Clarabel
  takes, and `A` is filled column by column in conic-row order. The delivered
  code sorted every entry in Clarabel's triplet constructor. Conversion time
  per run fell from about 0.145 to 0.075 s at 164 variables and from 0.023
  to 0.010 s at 84. A test checks bit equality with the triplet path on 300 random models
  with infinite, one-sided, equal and free sides and explicit zeros.
- The cache key is compared in place instead of being copied on every call.
- The BFGS update is computed entry by entry, without five n×n temporaries.
  A test checks bit identity with the matrix expression, in debug and release.
  BFGS time fell 5–10%; the rest of it is the Cholesky check.

Objectives and iteration counts are bit-identical to the delivery in both
modes. The 13 cases, the 1,200 stress problems and HS071 are unchanged.

The saving is about 2% of the measured work at 164 variables. In wall time it
is smaller than the effect of code layout. I built the delivered code with a
meaningless two-line change in `convert`, and it ran the unchanged Clarabel
engine 3–4% slower on the same inputs. The reviewed build ran it 5–6% slower,
with the same iteration counts. Both effects come from where the linker
places Clarabel's hot loops, not from the work done. On this host, wall-time
differences between builds below about 5% are therefore not meaningful;
compare phase timers, or cold and reuse within one build, as the delivered
benchmark does.

**Options measured but not adopted** (orbit with 164 variables unless noted):

| Option | Effect | Acceptance |
| --- | --- | --- |
| Clarabel's `faer-sparse` factorizer | 38% slower; factorization 1.76 s instead of 1.60 s, solves 1.87 s instead of 0.72 s, also single-threaded; adds about 70 crates | unchanged |
| Iterative refinement off | 17% faster (15% at 84 variables), same interior-point iterations | 1 of 1,200 stress problems becomes `qp_failure` (cold) |
| `qp_tolerance = 1e-8` (Clarabel's default) instead of `1e-10` | 12% fewer interior-point iterations (1,446 → 1,278), about 6% faster; 10% fewer on the stress problems | identical with reuse: 13/13, 1,186/1,200, no false claims |
| Reuse on by default | 15% faster on the orbit, 12–17% on 300 small stress problems | identical in both modes |

The defaults are unchanged, as the plan specifies. Reuse and the looser QP
tolerance are the two measured wins without an acceptance cost. Changing either
default is the owner's decision. Beyond them, the per-iteration cost is the
dense factorization. A material further speedup needs fewer QP iterations (a
warm-startable or active-set QP) or a structured or sparse Hessian. The plan
defers both. Each B is also factorized twice per iteration, once by the BFGS
check and once by the adapter's positive-definiteness guard (about 0.07 s
each at 164 variables). Removing one would need the QP contract to carry a
verified flag.

## Tests and limitations

**82 Rust tests plus one doctest (79 delivered, 3 from the review); 85
compiled Python tests; 42 dummy-only Python contract tests pass.** Formatting, Clippy with warnings denied,
Rust 1.85 compatibility, rustdoc with warnings denied and strict mypy also
pass. See [validation](validation.md) for commands and scope.

New cases cover complete numeric replacement, explicit zeros, cone/side and
pattern/dimension invalidation, dual reconstruction, cloning, cache disabling,
old cancellation tokens, in-flight cancellation, time/iteration caps, a real
partial update failure, forced acceptance rejection, retry iteration accounting
and cold confirmation of infeasibility. Compiled Python cases cover analytic
problems, the orbit, restoration, callback errors and limits, observer stop,
independent worker ownership and report defaults.

The dense 256-variable cap, one thread per solve, lack of automatic NLP scaling
or second-order correction, and local restoration failures remain. Reused
initial equilibration can worsen conditioning; the cold retry cannot guarantee
identical behavior, and a budget-limited reused attempt is not retried. No
application trajectory data, multi-core speedup or free-threaded Python was
validated. Cross-platform CI is configured but was not run in this workspace.
A useful next iteration would start from application data or the remaining
restoration failures, while keeping the new phase reports for performance work.
