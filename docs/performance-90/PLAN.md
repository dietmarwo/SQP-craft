# SQP-craft performance check and 30-iteration optimization plan

27 November 2026 · Baseline `d6b28ca62b88b7c92a31e556c408486fa66e28bc` · Rust workspace 0.20.0

**Start with the cost of solving the step QP.** On the 4,994-variable Earth–Mars transfer, 46.31 of 50.73 seconds went into the Clarabel engine. SQP built the engine workspace once, updated it 212 times and spent 0.43 seconds updating BFGS. Removing the entire BFGS update would save under 1% on this run. Finishing inside 30 seconds on this host needs roughly a 41% reduction in verified solve time. The first experiments therefore try to avoid full interior-point solves, reduce their dimension, or take fewer major steps.

**The clearest small-problem regression is HS38 with exact curvature.** The default solver verified all 20 problems in the frozen curvature panel. Exact curvature verified 19. On HS38 it shifted the Hessian 493 times and exhausted 500 iterations. On BQPGABIM the same exact mode converged in one iteration. These cases call for different experiments from the ones that speed up a dense factorization.

This is a new campaign. The repository's first 30 iterations optimized single-thread paths and its next 30 investigated explicit parallelism. The 30 proposed changes below continue that numbering at 61–90. The measurements in this section were executed for this plan. The numbered changes are proposals.

## Baseline measurements

### Native dense SQP

I built the pinned source with Rust 1.88.0, `--release --locked`, on an AMD EPYC 9V74 Linux guest. Each timed solve used one pinned logical CPU, full BFGS, an explicitly selected backend and workspace reuse. Each combination ran once for warmup and five times for measurement in fresh processes, rotating and reversing the order. The table gives median solve time and the observed range in milliseconds. The native example checks convergence, all four original-coordinate KKT residuals and the reference objective on these sizes.

| Fixture | Variables | SQP steps, both | Active-set ms [range] | Clarabel ms [range] | Clarabel / active-set median |
|---|---:|---:|---:|---:|---:|
| Orbit 20 | 84 | 104 | 31.00 [29.89–33.28] | 239.27 [232.92–243.94] | 7.72× |
| Orbit 40 | 164 | 167 | 270.27 [255.18–365.68] | 1,866.34 [1,829.20–1,907.16] | 6.91× |
| Separable 128 | 128 | 21 | 17.91 [17.14–19.15] | 117.43 [114.60–119.99] | 6.56× |
| Restoration 64 | 64 | 4 | 0.675 [0.66–0.74] | 5.24 [5.04–7.63] | 7.76× |

On Orbit 40 the active-set backend spent a median 251.16 ms inside the QP call, including 206.48 ms of engine time. The fixture made 168 QP calls. Its active-set counter reported zero charged inner iterations, although rebuilding factors and enforcing equalities still cost time. Clarabel made one setup and 167 updates. These timings compare explicit native backends. The public Python `auto` policy selects active-set with elastic steps for dense curvature and Clarabel for structured curvature.

### Earth–Mars multiple shooting

I ran the source-built Python extension with pykep-rust 0.1.6, partitioned BFGS, workspace reuse, the Lambert start, four throttle segments per arc, floor 0.02 and `scaling="none"`. One fresh process per row, one numerical thread. Solve time includes Python callbacks and excludes the independent post-check and DOP853 replay. These single observations locate costs and budget failures. They are unsuitable for ranking small timing differences.

| Arcs | Variables | Budget s | Solve s | Clarabel engine s | SQP steps | Interior-point iterations | Independent outcome |
|---:|---:|---:|---:|---:|---:|---:|---|
| 7 | 130 | 30 | 0.159 | 0.132 | 42 | 622 | Verified |
| 51 | 966 | 30 | 3.72 | 3.28 | 108 | 1,817 | Verified |
| 101 | 1,916 | 30 | 8.93 | 8.02 | 127 | 2,228 | Verified |
| 263 | 4,994 | 30 | 30.01 | 27.49 | 124 | 2,352 | Time limit, stationarity 1.19e−6 after multiplier recovery |
| 263 | 4,994 | 90 | 50.73 | 46.31 | 212 | 3,947 | Verified with supplied multipliers, stationarity 7.09e−7 |

The two 263-arc rows are independent cold starts. The short run was already primal-feasible at 9.59e−10. The extended run reached primal residual 3.08e−10, complementarity 2.75e−10 and zero dual-sign violation. Its global DOP853 position defect was 0.000223 km. Peak process RSS was about 112 MiB. At 101 arcs, removing all Python callback time would save at most 0.38 seconds on the observed 8.93-second solve. Callback work deserves its own benchmark with an expensive model.

### Small nonlinear problems

I ran both the public 0.20.0 wheel and an extension built from the pinned source through the same current S2MPJ harness. Each build ran 20 problems under `auto` and the same 20 under `clarabel+exact`, with a 500-iteration cap, 5,000 model calls and a 60-second solve limit. Both builds report version 0.20.0, so the binary identity matters. The wheel verified 20/20 and 18/20 respectively. The source verified 20/20 and 19/20. On both builds every strict convergence claim passed independent verification. The source already contains the zero-Hessian fix for CATENARY.

| Case on pinned source | Auto BFGS, steps | Clarabel exact, steps | Observed work and implications |
|---|---:|---:|---|
| HS38 | 27 | 500, rejected | Exact requested 500 Hessians and shifted 493 times. Final stationarity was 9.08e−3. |
| HS71 | 5 | 56 | Exact took 57 Hessians and 57 shifts. Both endpoints verified. |
| CATENARY | 67 | 20 | On the release wheel exact ended in `qp_failure`. Keep this as a guard for any shift change. |
| DTOC2 | 60 | 11 | Exact used 16 value, 12 derivative and 12 Hessian calls. Auto needed 61 value and 61 derivative calls. |
| BQPGABIM | 49 | 1 | Auto rejected 229 trial points, with 279 value and 50 derivative calls. Exact used two calls of each kind. |

These two panel arms change both curvature and backend. Use explicit Clarabel BFGS and Clarabel exact to isolate curvature, and the same curvature on each backend to isolate backend effects. Count Hessian work and compare objectives before interpreting any callback ratio.

### Other evidence used to select experiments

| Archived study | What it establishes | Use in this campaign |
|---|---|---|
| Historical 1,036-case S2MPJ first-step study | Automatic baseline 814 accepted. Step limit plus initial scaling reached 824, with 23 gains and 13 losses. The bound-constrained subset lost one net. | Retest the named gains and losses. Publishing a new default requires evidence across subsets. |
| EPIGRIDS, 250-bus AC dispatch | The best documented configuration verified 5/14 starts. Some remaining points meet power balance and cost targets but miss currency-unit stationarity. Reference starts are easier than source or DC starts. | Optimize difficult feasible endpoints. Keep source, DC replay and reference-start counts separate. |
| Four-branch spring–mass QP | The 3,822-variable fixture has a common six-variable input. Standard minimum-degree and shared-root-deferred orderings both gave 204,333 factor entries. Fixed reverse-stage order gave 240,081. | Test numerical branch solves and synchronization cost. The symbolic fill result alone establishes no speedup. |
| Earlier eight-thread campaign | Independent solves benefited from outer parallelism. A native propagation surrogate also benefited from parallel callbacks. The Clarabel engine in that build was serial. | Give batch throughput and single-solve latency separate experiments and budgets. |

The full 1,036-case suite, AC study, FanCyQPALM reference and eight-thread matrix were not retimed for this assessment. Historical wall times use different hosts or builds. [First-step records][first] · [EPIGRIDS][power] · [Fan QP records][fan] · [Parallel campaign][parallel]

## How to run a round

1. **Freeze the baseline.** Build the SHA above and make a separate worktree for candidates. Save the full revision, lockfile and binary hashes, CPU affinity/quota, compiler flags, dependency versions, starts, scaling, thread count, budgets and selection hashes. Pin pykep, NumPy, SciPy and BLAS kernels for differential runs. Keep one immutable baseline and one incumbent containing only retained changes.
2. **Measure one mechanism per round.** Before editing, record the hypothesis, source entry point, primary metric, adverse case and expected numerical behavior. Add phase counters in a profiling build where necessary. Run both implementations under the same instrumentation, then time release builds with diagnostic tracing off. Compare candidate to incumbent. At each group of five, rerun the cumulative candidate against the original baseline.
3. **Keep numerical and timing verdicts separate.** Record status, independently verified point, false claims, objective, source of multipliers, all four residuals, actual F/D/H attempts and time in each callback. Record SQP/QP iterations, setups, updates, cold retries, shifts, rejected trials, fill and peak RSS where applicable. Recheck from original model evaluations. A censored run counts against coverage. Exclude it from paired timing and retain its elapsed time and terminal residuals.
4. **Pair the measurements.** One warmup plus seven interleaved A/B rounds for targeted timings, same core and settings, fresh processes. Alternate A/B order. Report paired median speedup, individual samples and a paired 95% bootstrap interval. Repeat noisy wins on a second session. Time cheap sub-millisecond kernels in batches. Use three screen repeats for the 263-arc/AC cases and seven for promotion of a timing claim. Eight-thread tests reserve eight allowed CPUs for one timed child at a time.
5. **Make a decision.** The row-specific thresholds below are proposed promotion targets. For a speed change, require at least 5% targeted end-to-end improvement with a paired interval above 1.0, or a verified memory saving of at least 15% with comparable latency. A faster kernel alone can justify another experiment. For a convergence change, require a newly verified case or at least 10% lower targeted model work with matched objectives, and record time as well. Keep slower, negative and inconclusive results in the ledger. Revert rejected implementation changes before the next experiment.

**Numerical gates.** Preserve all public tolerance, work-limit, cancellation, memory and backend-selection contracts. For the native fixtures and SQP5 use absolute primal ≤1e−7, stationarity and complementarity ≤1e−6, dual-sign ≤1e−8. Use the repository's own grade for S2MPJ, including its gradient-scaled dual check and objective comparison. EPIGRIDS requires original currency-unit residuals and the independent PYPOWER cost and feasibility gates. If candidate and reference reach different accepted objectives, list both and exclude that pair from speed/work claims. HS2 is a standing example. Record strict status and recovered certificates separately. A false convergence claim rejects the change.

**Regression and holdout policy.** Use Orbit 20/40, Separable 128, Restoration 64, the 20-case curvature panel and SQP5 Lambert 7/13/26 for rapid tuning. Use 51/101 Lambert and 263 for scaling and deadlines. Reserve 51/101 offset starts, native sizes 30/48, 96/192 and 48/96 by family, and a deterministic held-out fraction of the remaining full S2MPJ selection for final confirmation. Fix those names before tuning, for example the first fifth by SHA-256 of the case name within each subset after removing development cases and all named regression cases. Keep the 53-case backend-disagreement panel and all first-step wins/losses as known regression cases. Freeze fresh RHS/initial-state seeds for the fan QP. Check each group of five against known sentinels; open the unseen split only for the final decision. For default promotion require zero new false claims, zero lost verified protection cases, no net loss in any full-suite subset, and explicit review of every individual loss. Investigate any reproducible >5% untargeted slowdown. A specialized method can remain opt-in when the wider default gates fail.

## Thirty distinct experiments

Each row names a change to implement, the control to beat and the result that would justify keeping it. The targets are deliberately testable. They are not forecasts. Numbers 61–90 correspond to rounds 1–30 of this plan.

### Rounds 1–5 · Reduce work in the structured SQP5 solve

| ID | Change and implementation point | Benchmark and decision |
|---:|---|---|
| 61 | **Predict the next active set.** In the Clarabel adapter, use the last accepted row/box sides to propose a solution of the *new* equality-constrained QP before calling the interior-point engine. Refactor its current numeric matrix, lift all duals and check every side in original units. Fall back to ordinary Clarabel under the remaining budget on any failure. | Replay consecutive SQP5 QPs, including changing activities. Compare prediction hit rate, wrong-side rejection, added cost on misses and total solve time. Target ≥10% faster at 51/101 arcs with identical verification. |
| 62 | **Condense pairs of shooting arcs.** Prototype an optional step-QP adapter that eliminates one interior state from every two neighboring dynamics blocks. Carry its bounds into reduced rows, retain constants and reconstruct primal and dual variables before the full QP check. Admit only a well-conditioned eligible block. | Compare full Clarabel with condensed Clarabel on the same captured matrices, then complete 26/51/101-arc solves. Record fill, condition failures and back-substitution cost. Keep if ≥10% faster end to end without a memory-cap breach. |
| 63 | **Scale rows and variables by arc block.** Extend `crates/sqp/src/scaling.rs` with bounded, fixed-per-solve scales for state continuity, thrust rows and variables. Apply inverse transformations to multipliers, and preserve the merit and checker in original units. | Compare against both `none` and existing Jacobian scaling on identical starts. Target recovery of a currently failing scaled 51/101-arc case and no loss of the unscaled quality benchmark. |
| 64 | **Initialize each partition’s curvature from its own first usable secant.** In `partitioned.rs`, test a clipped positive local spectral scale with overlap weights preserved. Skip tiny or adverse secants and keep current reset behavior. | Check disjoint blocks against the analytic update and overlapping arcs against current partitioned BFGS. Seek ≥10% fewer accepted steps or interior-point iterations on matched 26/51/101-arc solutions, with a net wall-time gain. |
| 65 | **Predict a multiplier correction near feasibility.** At a near-feasible iterate with persistent stationarity error, fit multipliers on active rows and bounds and use them to form the next step model. Only the ordinary original-unit termination test may certify success. Count this work and check for rank deficiency. | Test the 263-arc deadline and source/DC AC endpoints. Compare current multipliers, predicted multipliers and independent certificates. Keep if a previously missed deadline/start verifies or the matched large solve takes ≥10% fewer QPs. |

### Rounds 6–10 · Improve exact-curvature decisions

| ID | Change and implementation point | Benchmark and decision |
|---:|---|---|
| 66 | **Bracket the smallest usable scalar shift.** In `exact.rs`, use the existing sufficient shift as an upper bound and try a bounded positive-definiteness probe inside that interval. Reuse the symbolic pattern. Count probes and keep the CATENARY zero-Hessian floor. | First use frozen HS38 and HS71 Hessians, then nonlinear solves. Target HS38 verification inside 500 steps or ≥25% fewer HS71 Hessians. Reject a scheme whose extra factors erase the solve-time benefit on already-fast exact cases. |
| 67 | **Use a nonuniform diagonal correction.** Try modified Cholesky with local pivot corrections and bounded fill. Include the applied correction in the QP model and its diagnostics. Keep the checked scalar-shift path for breakdown. | Compare directly with 66 and the original shift at equal budgets. Test indefinite, zero and nearly singular curvature, HS38, SQP4 and the kinetics fit. Retain if an extra case verifies or H calls fall ≥15% without degrading the positive-definite controls. |
| 68 | **Add curvature only in equality-normal directions.** For eligible equality rows, add ½ρ‖Ap−b‖² to the step objective. Include the linear term −ρAᵀb, cap fill and reconstruct/check multipliers against the original model. | Check analytic equality QPs, redundant rows, HS71 and SQP4. Record tangent-step error, added fill, shifts and H calls. Seek ≥20% fewer H calls or steps on a shifted case with the same independently verified objective. |
| 69 | **Use a temporary BFGS model after repeated damaging shifts.** Drive a bounded exact-to-quasi-Newton transition from observed shifts and poor predicted/actual reduction. Reset stale curvature on switching and report which model produced each step. | Compare to always-exact and always-BFGS on HS38, HS71, CATENARY, DTOC2 and BQPGABIM. Target 20/20 verified in the hybrid arm while retaining most of exact mode's work savings on DTOC2 and BQPGABIM. Keep the hybrid explicit. |
| 70 | **Select the backend for small exact Hessians by structure.** Add an opt-in dispatch rule to the existing backend resolver. Test active-set on supported small dense exact models and retain Clarabel for sparse/large models. Freeze the crossover on development cases and honor explicit backend requests. | First run the 20 cases as a two-by-two study, full BFGS/exact × active-set/Clarabel, with elastic mode off. Then test the frozen dispatch against always-Clarabel and always-active-set where supported. Target ≥10% lower paired exact-mode time versus current auto on common accepted objectives, with no lost verified case. |

### Rounds 11–15 · Lower dense active-set QP latency

| ID | Change and implementation point | Benchmark and decision |
|---:|---|---|
| 71 | **Factor dense Hessians in panels.** In `crates/activeset/src/dense.rs`, test a blocked Cholesky with panels of 8, 16 or 32 and contiguous trailing updates. Select a crossover on development sizes. Keep the current path for small or diagonal matrices. | Use changing-H QPs at 84/128/164/256 and Orbit 20/40. Check pivots and residuals on SPD, semidefinite and indefinite matrices. Target ≥10% lower factor time and ≥5% lower native solve time. |
| 72 | **Solve triangular systems for multiple inverse columns together.** Tile the computation of L⁻ᵀ to reuse each factor panel across RHS columns. Preserve the diagonal shortcut and measure one thread first. | Compare with `inverse_transpose` on changing, identical and barely changed Hessians. Report inverse residual and complete QP time. Require ≥10% less inverse time and a measurable end-to-end win on Orbit or Separable. |
| 73 | **Solve equality-dominated QPs directly.** Form a range-space system from H and the equality rows. Accept its proposal only if all omitted box and inequality sides and their multipliers pass. Return to the existing working-set path for dependence or violated inequalities. | Orbit's 105/168 QP calls are the primary cases. Include duplicated equalities and cases with active bounds. Target ≥10% faster Orbit 40 under the current limits; record fallback rate on mixed constraints. |
| 74 | **Update the factor after a known BFGS rank-two change.** Pass accepted update vectors through a private SQP/backend path. Attempt a stable Cholesky update followed by downdate. Invalidate on any unrelated H, scale change, reset, loss of definiteness or excessive residual. | Compare against fresh factorization on long changing-H sequences, alternating resets and both Orbit fixtures. Record refactor frequency, backward error and total time. Target ≥10% faster on at least one dense fixture and unchanged numerical acceptance throughout. |
| 75 | **Eliminate exactly fixed coordinates before a dense QP.** Substitute the fixed step into H, gradient, rows and bounds. Lift primal variables and bound multipliers using stationarity before checking the original problem. Cache the index map only while the fixed set is unchanged. | Use BQPGABIM, DTOC1L/DTOC2, all-fixed, changing-fixed-set and no-fixed controls. Target ≥10% lower QP time on fixed-heavy cases, valid supplied duals and no penalty exceeding noise on the no-fixed path. |

### Rounds 16–20 · Spend fewer callbacks on unsuccessful steps

| ID | Change and implementation point | Benchmark and decision |
|---:|---|---|
| 76 | **Interpolate the backtracking step.** After a finite rejection, compute a safeguarded quadratic trial length from the merit value and initial slope. Fall back to the existing halving rule for undefined trials or an unusable interpolant. | BQPGABIM auto currently rejects 229 trials. Target ≥25% fewer actual function calls there, plus a net reduction on the high-rejection slice. Test domain-limited models and preserve the ordering of value and derivative callbacks. |
| 77 | **Trigger the first-step safeguard from evidence of failure.** Apply bounded initial curvature rescaling only after observing a rejected first trial and an unusually long scaled step. Use fixed, problem-independent thresholds and retain the accepted-step path. | Replay all 23 gains and 13 losses of the previous safeguard against unchanged `auto`. Gate on net verified gain on the full suite, no net subset loss and no increase above 2% in median paired actual calls. Never recognize cases by name in the solver. |
| 78 | **Adapt the elastic penalty after observing slack progress.** In the active-set SQP driver, adjust the growth of the penalty to residual reduction and prior dual magnitude. Limit repeated increases and retain hard linear rows. | Use infeasible linearizations, mixed hard/soft rows, HS25NE and feasible hard-row controls. Target ≥10% fewer restoration/elastic QPs or an extra certified solution. Audit local-infeasibility and limit statuses independently. |
| 79 | **Choose a restoration trust bound from model agreement.** In `restoration.rs`, update its radius from predicted versus actual decrease in row residuals, with bounded shrink/grow factors. Respect the remaining restoration budget. | Run restoration 64, UBH5 and the EPIGRIDS base/tight cases. Track steps back to ordinary SQP and time to verification. Keep for ≥1 newly verified start or ≥15% less restoration work on baseline-verified cases without a lost success. |
| 80 | **Try a small feasibility/objective filter after repeated merit rejections.** Track a bounded set of accepted pairs. Use a sufficient-progress margin and a restoration fallback on the same budgets. | Stress Maratos/SOC cases, EPIGRIDS source/DC starts and the ordinary 20-case panel. Target at least one additional AC source/DC verification or ≥15% fewer callbacks on the rejection-heavy subset. Record cycling and different local objectives. |

### Rounds 21–25 · Reduce sparse linear-algebra and interface overhead

| ID | Change and implementation point | Benchmark and decision |
|---:|---|---|
| 81 | **Precompute sparse Cholesky destination slots.** In `crates/core/src/sparse.rs`, cache the numeric scatter from input entries to factor entries. Separately time a bounded lookup table for the repeated inner-loop slot searches. Charge table storage before allocation. | Test narrow arc bands and high-fill adverse graphs. Compare factor decisions and reconstruction against the current routine across changed values and patterns. Seek ≥20% less PD-check time; use its measured share to justify adoption. |
| 82 | **Use fill-aware memory admission for polishing.** In `crates/clarabel/src/polish.rs`, replace the dense worst-case reservation with a conservative, budget-capped symbolic bound for the actual reduced pattern. Abort safely on predicted excess or symbolic work limit. | Replay both captured EPIGRIDS QPs and growing sparse KKT fixtures deliberately crossing the old gate. Keep if a previously excluded case passes its original QP check within the memory limit. Measure peak allocations as well as the estimate. |
| 83 | **Retain the polish pattern when active sides repeat.** Cache the reduced permutations, row/column scatter and symbolic analysis across correction attempts or consecutive QPs with an identical signature. Refresh all numerical values and RHS data. | Alternate repeated signatures, single-side flips, a removed fixed bound and rank loss. Record symbolic versus numeric time and checked fallbacks. Target ≥15% less polish time and ≥5% less time on the affected end-to-end AC solve. |
| 84 | **Reuse the restoration Gauss–Newton assembly map.** Precompute the lower-CSR slot of each eligible Jacobian pair in `restoration.rs`. Retain the workspace across retries, respecting the structured memory budget. Recompute active violations and numerical contributions each time. | Compare the assembled matrix and gradient with the current binary-search path on sparse and overlapping rows. Seek ≥20% lower assembly time on restoration-heavy AC or structured stress and a verified end-to-end improvement. |
| 85 | **Bulk-copy Python callback outputs.** In `crates/python/src/lib.rs`, add a fast path for contiguous native-endian binary64 buffers for Jacobians, gradients and Hessians. Keep arbitrary iterables, strides, size errors and Python exceptions on the existing path. Use a model that returns arrays directly. | Measure binding-only cost on empty/cheap callbacks, then run the same NumPy models through both paths on SQP4 and SQP5. Include noncontiguous/foreign-endian arrays, generators and short/long outputs. Target ≥20% lower conversion cost and ≥5% total time on a binding-limited workload. |

### Rounds 26–30 · Test structural and parallel changes against full solves

| ID | Change and implementation point | Benchmark and decision |
|---:|---|---|
| 86 | **Recover sparse Hessian entries from analytic Hessian-vector products.** Prototype graph colouring on a declared sparsity pattern, first on fixtures with both analytic H and HVP. Recover the required triangle, preserving multiplier signs and diagonal entries. | Compare HVP count, colouring overhead, memory and total callback time with the existing exact callback and its numerical derivative check. Keep the API prototype only for ≥20% less Hessian time on genuinely sparse, nontrivial models and matched convergence. |
| 87 | **Compute partitioned element updates concurrently.** Use the existing bounded execution resources for independent local updates, then scatter contributions in the current fixed order. Set a size threshold from one-thread measurements. | Try 1/2/4/8 threads on large/clique-heavy models and 51/101 arcs. Check deterministic matrices, resets and solutions. Retain only if a complete target solve speeds up ≥5%; the small measured BFGS share on the standard transfer makes a kernel-only win insufficient. |
| 88 | **Solve branch KKT systems independently around the shared root.** Prototype one Newton-system kernel on the published fan QP. Factor each branch, accumulate the root Schur complement in fixed order, solve the six-variable root, and back-substitute. Use one precision and count per-worker scratch. | Compare with a monolithic solve of the *same* linear system and with a complete checked Clarabel QP. Test fresh seeds plus 1/4/8 workers, rank loss and a case without a shared root. A kernel win must survive integration and yield ≥10% checked QP speedup. |
| 89 | **Reuse an immutable structural analysis between independent solves.** For repeated models, share only validated sparsity, scatter maps and budgeted symbolic metadata. Each solve owns numerical factors, multipliers, cancellation, and model state. Test the normal worker/batch entry point. | Measure cold first solve and amortized time for 16 solves of the same layout. Mutate row kinds or one structural entry at a time. Target ≥15% less setup or memory and ≥5% more checked solves per second on a setup-heavy workload. |
| 90 | **Schedule mixed-cost batches by estimated cost.** Extend the existing dynamic batch assignment with an opt-in longest-estimated-first order from declared size or earlier independent timings. Preserve input-order results, per-start budgets and worker-local backends. | Compare current dynamic/input order with cost order at outer×inner 8×1, 4×2 and 1×8 on mixed orbit and propagation batches. Include wrong estimates and cancellations. Target ≥10% lower makespan at fixed work without loss of checked results. Report p95 latency as well as throughput. |

**Managing larger experiments.** For condensing, normal-space augmentation, HVP recovery and branch elimination, begin with a captured QP or analytic model. Stop that prototype if it cannot reconstruct the original residuals, exceeds its memory budget, or has too little headroom to reach its stated target. Record what was implemented and measured. Continue the next distinct experiment from the last accepted incumbent. For interacting winners, run baseline, A, B and A+B before combining them.

## Reproduction and final decision

From the pinned checkout, in an activated Python environment with NumPy 2.3.5, SciPy 1.17.0, pykep-rust 0.1.6 and maturin installed. Choose an allowed core in place of CPU 0 if necessary. Keep outputs from different bindings and different policies in separate directories. The S2MPJ sources are pinned by the repository harness.

```bash
git checkout --detach d6b28ca62b88b7c92a31e556c408486fa66e28bc
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo build --release --locked -p sqp-craft-solver --example performance
python -m maturin develop --release --locked
python -m bench.s2mpj fetch
mkdir -p run-base
python -c 'import sqp_craft._native as m; print(m.__file__)'
taskset -c 0 python -m bench.run docs/benchmarks/pounce-curvature/selection.json \
  run-base/panel.jsonl --evidence --jobs 1 --solvers auto,clarabel+exact
taskset -c 0 python -m bench.trajectory_scale --arcs 263 --method partitioned \
  --scaling none --floor 0.02 --seconds 30 --output run-base/263-30.json
taskset -c 0 python -m bench.trajectory_scale --arcs 263 --method partitioned \
  --scaling none --floor 0.02 --seconds 90 --output run-base/263-90.json
```

For the native table, run `target/release/examples/performance FAMILY SIZE reuse 0 BACKEND` on one core. Use the four family/size pairs in the table crossed with `active-set` and `clarabel`, one warmup and five measured rounds. For the campaign, the repository supplies the fuller paired runner below, including both cold and reused workspaces. Set `BASE` and `CAND` to distinct absolute binary paths first.

```bash
taskset -c 0 python scripts/compare_performance.py run-base/paired.json \
  "base-as=$BASE:active-set" "candidate-as=$CAND:active-set" \
  "base-cl=$BASE:clarabel" "candidate-cl=$CAND:clarabel" --repeats 7
```

Baseline binary SHA-256 values were `0221c496aea50f3304a44fc9a2ce8d8076c4061d2a7c1b337c446ed0638f7d78` for the native example and `b2f5586494c88b7aa2e917b243cb8d5b88c9982274b73dc617f89d3703b86448` for the source-built Python extension. The public wheel extension hashed to `7c89d508dcada2a7232ffb381de26779a0ccf38fbfa28c467b4a1d4f45a35fe5`. The S2MPJ revision was `fea6a70048eaad28b13a08703ddbfdbf65cd9c30`. CPython was 3.12.14. Use these to identify this run. Compiler, build path and packaging differences can change the hashes on a rebuild.

**Final validation.** Run workspace Rust tests with `--locked`, the Python suite against the candidate extension, the shared job tutorial and numerical tutorials. Replay adverse cases for backend changes, cancellation, partly updated workspaces, fixed variables, wrong active sets and ill-conditioned matrices. Repeat the baseline and final candidate on the full 1,036-case selection and the 14 AC starts with identical budgets. Recheck SQP5 on offset starts and both deadlines. Run the 20-case POUNCE comparison as an external control for curvature changes. Report per-case gains/losses, any changes of basin, median and tail timing, and single-solve versus batch gains. Promote a change to the public default only where the wider evidence supports it.

**The execution log for the campaign** should have one row per attempted change with parent and candidate SHA, exact commands, raw result paths, target/control/holdout, verification counts, F/D/H, time with dispersion, memory, decision and reason. Archive rejected experiments separately from the clean patch. A useful outcome may be a faster current default, a checked optional specialization, or evidence for abandoning an approach.

## Source index

[Source tree][tree] · [Native benchmark and its checks][native] · [Backend policy][policy] · [Prior single-thread results][serial] · [Prior eight-thread results][parallel] · [Curvature panel and its binary provenance][curvature] · [Structured modes][structured] · [SQP5 physical model and limits][target] · [First-step records][first] · [AC dispatch study][power] · [Fan QP fixture][fan]

[tree]: https://github.com/dietmarwo/SQP-craft/tree/d6b28ca62b88b7c92a31e556c408486fa66e28bc
[native]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/crates/sqp/examples/performance.rs
[policy]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/docs/backend-selection.md
[serial]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/docs/performance-30/RESULTS.md
[parallel]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/docs/performance-60/RESULTS.md
[curvature]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/docs/benchmarks/pounce-curvature/README.md
[structured]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/docs/structured-sqp.md
[target]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/docs/target-card.md
[first]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/docs/benchmarks/first-step/tables.md
[power]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/bench/epigrids/README.md
[fan]: https://github.com/dietmarwo/SQP-craft/blob/d6b28ca62b88b7c92a31e556c408486fa66e28bc/bench/fan-qp/README.md
