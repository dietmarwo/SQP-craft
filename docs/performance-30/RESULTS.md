# SQP-craft 0.15.0: thirty autonomous performance iterations

Completed **30 planned, implemented and tested iterations** against original main `4d776bbd89f187fe176be805ab91fd64305a8c44` from https://github.com/dietmarwo/SQP-craft. Every iteration has a pre-implementation plan, source checkpoint, release Rust test run and paired measurements. The release adds a final two-entry CSR refinement after the comprehensive kernel audit found a regression. It reached `main` together with 0.16; see [REPRODUCE.md](REPRODUCE.md) for where the checkpoint history is.

The active-set path improves on all four primary SQP fixtures and all six declared held-out sizes. The default Clarabel backend has mixed wall-time results. The evidence does **not** support a claim that every individual iteration or every workload is monotonically faster. Defaults, callback formulas, mathematical problems, tolerances and budgets were preserved.

## Primary end-to-end result

One pinned numerical thread, one warmup plus seven measured fresh-process repetitions, identical final harness on original/final source. No build, installation or stress process ran concurrently. Medians in milliseconds; speedup is baseline/final.

| Existing native fixture | Baseline 0.14 ms | Final 0.15 ms | Speedup |
|---|---:|---:|---:|
| orbit 20 | 34.473 | 28.882 | 1.194× |
| orbit 40 | 282.943 | 253.078 | 1.118× |
| separable 128 | 19.428 | 16.139 | 1.204× |
| restoration 64 | 0.982 | 0.653 | 1.504× |

The four-fixture geometric speedup is **1.247×** (about 19.8% less elapsed time by that geometric measure). Full-BFGS outer iteration counts remain **104 / 167 / 21 / 4**. The orbit fixtures contain 84/164 variables; they retain the same dynamics, initial point and reference-objective checks. Solver timing excludes native fixture construction and process startup.

All six held-out sizes were frozen before tuning: orbit 30/48, separable 96/192 and restoration 48/96. Every measured run converged and passed the same original-unit KKT assertions. Active-set speedups in the three-repeat campaign are 1.171× / 1.150× / 1.196× / 1.187× / 1.452× / 1.308×. For orbit 48 (196 variables), the median falls from 496.66 to 431.83 ms. Complete per-case samples and hashes are in data/holdout-native.json.gz and TABLES.md.

## Default backend and remaining regressions

Clarabel remains the default. In the five-repeat cold/reuse comparison, reused Clarabel orbit-20 is **230.13 → 244.24 ms** and orbit-40 **1762.72 → 1809.34 ms**; separable/restoration improve modestly. This is not a uniform default-backend speedup. The Clarabel numerical engine is unchanged, and BFGS phase time falls substantially (orbit-40: 16.25 → 7.66 ms), but the engine dominates wall time. These observations do not isolate code-layout effects from shared-host variability.

The three-repeat held-out Clarabel campaign includes separable-192 **297.27 → 302.15 ms** and restoration-48 **2.93 → 3.31 ms**. A targeted nine-repeat confirmation gives **305.13 → 298.54 ms** and **2.993 → 3.019 ms** respectively. Both campaigns remain available; the slower samples were not replaced. Do not infer a broad Clarabel win from the active-set results. The legacy cold/reuse comparator is unpinned; primary/common and held-out drivers pin each child to one allowed CPU.

The final two-entry CSR kernel remains about 4% slower than baseline in the primary sample (0.467 → 0.487 ms for 5000 complete products), after the release refinement recovered a 1.59× gain over the regressive candidate. Dense inverse/Cholesky micro timings also show small losses/noise. No universal microkernel speedup is claimed.

## Where the work was removed

- **01–04:** sorted-CSR endpoint checks, lower-triangle endpoint validation, reuse of already validated bounds, and one shared independently checked QP/KKT arithmetic path. Public input validation remains in place.
- **05–09:** row-local symmetric Hessian products, contiguous full-row products, short-row specializations and finite-data zero-multiplier skips. The Hessian-product fixed-work kernel improves about fourfold; callbacks are unchanged.
- **10–13:** fuse finite checks into BFGS triangle construction, use contiguous arithmetic before mirroring, stop negligible-step scans early, and copy exact-curvature differences without damping operations. The strict matrix-expression and padded-indefiniteness regressions remain. In the final cold/reuse comparison, active-set orbit-40 BFGS time falls **14.31 → 5.63 ms**.
- **14–20:** avoid unused factor clearing and row norms, project singleton normals directly, skip impossible insertions after basis saturation, restrict exact-zero reflector support, and construct exactly diagonal inverses directly. No small nonzero coefficient is dropped.
- **21–26:** skip zero KKT combination columns, compute activity magnitudes only where used, borrow nonelastic gradients, reserve retained working-set capacity and rebuild bounded/equality source lists once per model.
- **27–30:** checked unconstrained/diagonal-box QP proposals; complete bitwise Hessian keys for factor reuse; and an unrotated inverse cache promoted only after confirmed cross-call Hessian reuse. Changed H invalidates the caches; changed rows/bounds are processed afresh. Final acceptance never trusts a cache or heuristic claim alone.

PLAN.md indexes the detailed contract. Each `iteration-NN-plan.md` and `iteration-NN-results.md` records its own implementation/checks/measurements. VALIDITY.md gives the numerical arguments and cache boundaries. Source tags `sqp-perf30-00` through `sqp-perf30-30` preserve the sequence; the final release also includes the short-row guard and integration fixes.

## Repeated QP results are a separate workload

Each row below is **30 calls on one backend with a repeated Hessian**, including initial misses/promotion. These are total fixed-work batch times, not one cold QP or a trajectory speedup.

| Fixed QP batch | Baseline ms | Final ms | Speedup |
|---|---:|---:|---:|
| qp-dense 64 | 3.704 | 1.766 | 2.097× |
| qp-pairs 64 | 1.785 | 0.628 | 2.840× |
| qp-free-dense 128 | 11.578 | 1.964 | 5.895× |
| qp-box 128 | 6.693 | 0.196 | 34.223× |
| qp-zero 64 | 1.732 | 0.130 | 13.314× |

The large gains rely on structure or repeated H. Full SQP usually changes H, so it does not inherit these factors. For nonrepeating H, the inverse cache avoids an unused copy. Retained memory increases by an owned Hessian key and, after confirmed reuse, one dense unrotated inverse. All memory is backend-local; no shared global cache or new numerical dependency is introduced.

## Validation of the delivered code

| Check | Outcome |
|---|---|
| Rust | 173 regular workspace tests passed in debug/release; full release checks ran at every iteration; all-target/all-feature and warnings-denied Clippy checks passed. |
| Python | All 135 tests pass through the compiled extension, including original point, multiplier, callback, cancellation and tolerance assertions. |
| Paired independent NLP audit | **17,372 pairs / 34,744 calls**, with no lost convergence/acceptance/reference result, no changed outer iteration count and maximum audited accepted-objective difference **0.0**. No worker exception or rejected convergence claim. |
| Seeded random/elastic subset | 11,700 calls per version; 10,966 converged per version. Existing nonconverged outcomes remain visible. |
| Options/families | Baseline, Maratos, linear and MPCC families under default/SOC/augmented-curvature/Jacobian-scaling configurations; independent NumPy checks of development and held-out fixture sizes. |
| Harder QP census | 1000 cases, seed 2026, up to 100 variables; baseline/final outcome tallies match, including four active-set independent-check rejections and one limit. Those are not counted as successful solves. |
| Input integrity | Original callback/generator files match the pinned base byte for byte. Audit requires every frozen file, complete case/mode keys, correct versions and all 17,372 pairs. |

The solver-call total counts repeated configurations/versions, not unique mathematical problems. Independent NumPy checks require the solver's own multipliers to pass, rather than replacing a failed solver certificate with fitted multipliers. Random problems have no proven global optimum; numerical first-order acceptance is the claim.

## Failure and measurement ledger

1. The closed-form box path made old exact work-counter assertions obsolete. Two active-set tests and the Rust/Python SQP warm-start tests now assert zero changes for the first analytic QP and reuse of its returned active bounds (cold 6 changes versus warm 0). Point/multiplier/tolerance checks were retained; the Python test additionally compares outer iterations and callback counts. Failed first logs are included.
2. Iteration 17 initially showed a 12% orbit-40 regression; a nine-repeat comparison reduced it to under 1%. No reliable end-to-end gain is attributed to that tiny specialization.
3. The initial unconstrained-QP micro fixture was diagonal, obscuring iteration 27 after the diagonal inverse optimization. Added coupled-free coverage and rebuilt both numerical checkpoints with that same harness: iteration 27 measures **2.279×** there. Iteration 16 similarly added missing free-row/fixed/dense-normal coverage.
4. Final auditing found a large two-entry CSR dispatch regression. The release adds a direct two-term sum under the same arithmetic order, reruns numerical suites and all paired independent checks, and retimes baseline/final. `*-before-shortrow.json` preserves the slower candidate; `shortrow-confirmation.json` records the correction. A small residual micro difference remains as reported above.
5. Forcing a PyO3 library test with `--all-targets` first hit the runtime's relocated Python linker prefix (`/install/lib`). A private link to the real versioned library and an explicit native search path resolved that environment-only failure (the bundled unversioned symlink also pointed to a removed temporary directory). The original failure and successful all-target log are both retained; no solver code was changed for it.

## Scope and delivery

The existing low-thrust collocation fixture is a development surrogate; the actual user trajectory application remains unavailable. The 256-variable NLP cap, dense full Hessian and single numerical thread remain. No second-order/global nonlinear certificate or industrial-robustness claim is added. Default Clarabel and full BFGS remain unchanged; active-set is selected explicitly with `SqpOptions(qp_solver="active-set")`.

The archive includes source/tests, all 30 plans/results, baseline patch, source-history bundle, raw measurements, independent audit/stress dumps, validation logs, the checkpoint figure and file checksums. REPRODUCE.md documents common-harness builds, serial timings, complete paired validation and table/figure regeneration. TABLES.md contains the full measurements, including slower/noisy observations. The 31-checkpoint chart is based on one shared retiming campaign, not multiplied adjacent speedups; its 0.14 checkpoints and the separately measured 0.15 release should not be conflated.
