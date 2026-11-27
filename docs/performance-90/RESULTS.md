# SQP-craft performance campaign, iterations 61–90

27 November 2026 · SQP-craft 0.20.0 · Base `d6b28ca62b88b7c92a31e556c408486fa66e28bc`

The campaign implemented and measured 30 changes. Four earned a place in the patch. All four are opt-in. Three target structured or exact curvature. The fourth solves several dense active-set inverse columns per panel at moderate sizes.

| Iteration | Retained change | How to use it |
|---:|---|---|
| 63 | Bounded scaling within declared constraint elements | `SqpOptions(hessian="partitioned", scaling="elements")` |
| 64 | Local spectral initialization of overlapping BFGS elements | `SqpOptions(hessian="partitioned", partitioned_initial_scaling=True)` in Python, or the corresponding Rust or partitioned job setting |
| 66 | Bounded scalar shift search for small rejected exact Hessians | `SqpOptions(hessian="exact", exact_shift_bracketing=True)` |
| 72 | Eight-column panel traversal of the dense inverse factor | `SqpOptions(tiled_dense_inverse=True)` or `ActiveSetQp::default().with_tiled_inverse(true)` |

## Transfer results

I paired the same 0.02-throttle-floor Earth–Mars model, starts, one pinned numerical CPU, and original verification for partitioned BFGS with and without local spectral scaling. Timings are median wall seconds over verified pairs, after warmup. The host also ran the broader campaign. Absolute times varied substantially in this session. The stable step counts and the paired time ratios are the stronger comparison.

| SQP5 arcs and start | Paired repeats | Baseline → local steps | Baseline → local seconds | Paired speedup |
|---|---:|---:|---:|---:|
| 26 lambert | 7 | 75 → 40 | 1.59 → 0.89 | 1.87× |
| 51 lambert | 7 | 108 → 57 | 10.92 → 4.99 | 2.00× |
| 101 lambert | 7 | 127 → 78 | 15.03 → 9.82 | 1.59× |
| 51 offset | 3 | 110 → 58 | 6.52 → 2.89 | 2.37× |
| 101 offset | 3 | 125 → 78 | 21.20 → 9.13 | 2.01× |

Across the cases in this table, the iteration reduction survived the held-out offset starts. Point checks used fresh model evaluations in original nondimensional coordinates. The paired samples also checked arrival mass, and one measured pair per workload received a DOP853 replay. The code resets an element before spectral initialization if prior damped updates have already populated its off-diagonal entries. This is the reason the final 51-arc path took 57 steps instead of 62 in the original probe.

| 4,994 variables, Lambert start | Baseline with supplied KKT and convergence | Local spectral with supplied KKT and convergence |
|---|---:|---:|
| 30 seconds | 0/1 | 0/1 |
| 90 seconds | 1/3 | 2/3 |

The fastest verified local-spectral sample took 102 steps and 68.43 seconds. Time-limited runs are included in the counts above. The archived plan baseline was 212 steps in 50.73 seconds in a quieter session.
The only pair that both finished took 58.99 and 68.43 seconds, respectively. The campaign establishes a reduction in solver steps on the large case. The 30-second acceptance target remains open.

The 966- and 1,916-variable Lambert runs with standard Jacobian scaling ended in `qp_failure` at steps 6 and 7. Scaling by element enabled both to verify at 159 and 184 steps. The unscaled controls took 108 and 127 steps. The recovered scaled paths had final stationarity of 7.76e−7 and 6.95e−7, and DOP853 arc replay agreed with native defects to within 3.5e−15 in model units. Combining element scaling with local spectral initialization verified both cases in 85 and 113 steps.

## Exact and dense results

Bracketing raised independent exact-mode coverage from **19/20 to 20/20** in each of seven measured rounds. HS38 went from the 500-step cap to **65 steps and 66 Hessian calls**. HS71 and DTOC2 used about one-third fewer Hessian calls. CATENARY still verified and took two additional steps.

| Case | Plain → bracketed steps | Plain → bracketed F/D/H | Paired time ratio when both verified |
|---|---:|---|---:|
| HS38 | 500 → 65 | 501/501/500 → 74/66/66 | censored |
| HS71 | 56 → 38 | 57/57/57 → 39/39/39 | 2.03× |
| DTOC2 | 11 → 7 | 16/12/12 → 10/8/8 | 1.63× |
| CATENARY | 20 → 22 | 21/21/21 → 24/23/23 | 0.87× |
| BQPGABIM | 1 → 1 | 2/2/2 → 2/2/2 | 0.84× |

The patch's dense inverse candidate preserves within-column arithmetic. A native unit test compared all entries bit for bit at 80, 81, 164, 196, 255 and 256 variables, including partial panels. All ten native fixture sizes passed the existing endpoint assertions. Paired times below compare the rebuilt tiled candidate with the clean baseline. The measured candidate enabled tiling automatically. The delivered code exposes that same kernel through explicit opt-in. Child CPU counts setup and the fixture's post-check as well as the timed solve. The timings for unchanged short paths provide a measure of host and build variation.

| Native active-set fixture | Baseline ms | Tiled ms | Paired wall ratio | Paired CPU ratio |
|---|---:|---:|---:|---:|
| Orbit 20 | 110.87 | 106.39 | 1.36× | 1.22× |
| Orbit 40 | 1043.29 | 938.65 | 1.05× | 1.03× |
| Separable 128 | 74.38 | 68.40 | 1.09× | 1.09× |
| Orbit 30 | 456.91 | 452.20 | 1.07× | 1.13× |
| Orbit 48 | 1795.64 | 1630.78 | 0.91× | 1.05× |
| Separable 96 | 36.41 | 31.67 | 1.52× | 1.22× |
| Separable 192 | 158.98 | 143.62 | 0.97× | 1.00× |
| Restoration 96 | 1.64 | 1.62 | 1.09× | 1.04× |

The same-binary Orbit 20 screen gave a paired 1.135× speedup, with a 95% bootstrap interval of [1.025, 1.263]. The production rebuild's Orbit 48 wall median was lower for the baseline, while child CPU favoured the candidate. The intervals for both measures included parity on this fixture. All samples and intervals are in `evidence/campaign-summary.json`.

## Decisions across all 30 rounds

Each `Retain` appears in the production patch. All other rows have reproductions in the probe diff or in the archive's numerical replay scripts. A successful isolated kernel alone was insufficient where the change increased checked solve time, had little coverage on a complete solve, or required disproportionate infrastructure.

| ID | Change and verified observation | Decision |
|---:|---|---|
| 61 | Previous active sides proposed a new original-coordinate QP certificate. The 51-arc initial replay accepted 9 of 33 predictions across three measured passes. The live Rust path added work on the 101-arc case, with and without retention of the polish pattern. | Reject. The checked misses cost more than the hits saved. |
| 62 | Eliminating alternating 7-state shooting nodes cut the 51-arc QP from 966 to 791 variables. The original and lifted candidates passed all 36 initial replay checks. Combined H and A storage grew from 41,382 to 44,740 entries in the median replay. Building and solving the reduced QP took longer. | Reject at QP replay. An outer solve would have had a slower subproblem. |
| 63 | Pooled row/column scaling recovered both failing Jacobian-scaled Lambert tests, with original-unit residuals and trajectory replay. | Retain as `scaling="elements"`. Requires partitioned curvature and declared groups. |
| 64 | Local spectral BFGS initialization reduced steps consistently across development and offset starts. It gives the largest observed full-solve gain. | Retain as `partitioned_initial_scaling`. |
| 65 | Six projected coordinate sweeps improved some near-feasible multiplier estimates. The 101-arc run retained 127 steps, the 263-arc 30-second case still failed its check, and Winter12Generous lost its verified reference start. | Reject. The next curvature model became less reliable on AC. |
| 66 | Seven bisection probes and a final Cholesky check searched below the sufficient shift when a small nonzero exact Hessian was rejected. HS38 moved from 500 capped steps to 65 verified. The proximal zero-Hessian check and original work budgets still apply. | Retain as `exact_shift_bracketing`. |
| 67 | Nonuniform modified Cholesky verified HS38 in 38 steps and the entire pilot, but HS71 rose from 56 to 137 steps and required 138 Hessian calls. | Reject for mixed work and the HS71 regression. |
| 68 | Equality-normal augmentation with the linear term preserved analytic solutions, including redundant equality rows. Three later HS71 QPs passed. Its first Hessian had negative tangent curvature, and the tested SQP4 models remained indefinite while fill increased. | Reject at captured QPs. |
| 69 | A shift-triggered switch to BFGS verified all 20 and HS38 in 74 steps with 15 Hessians. HS71 reached 17 steps. CATENARY took 37 steps and 98 value calls. | Archive the hybrid. Bracketing preserved exact mode and took 22 CATENARY steps. |
| 70 | Two-by-two curvature/backend control and a size-based resolver chose active-set on the dense exact pilot. Both exact backends still missed HS38. The size predicate was non-discriminating across the small panel. | Reject this dispatch rule. |
| 71 | Blocked Cholesky updated trailing columns in panel order. Some separable calls improved, while both measured Orbit medians slowed in the screen. | Reject. The win failed to transfer to the target Orbit workload. |
| 72 | Inverse traversal kept one factor column hot across RHS columns and preserved each dot product. The repeat comparison covered ten native sizes with identical step counts. | Retain as `tiled_dense_inverse`. Timeouts in the broad sweep prevented default promotion. |
| 73 | Range-space equality proposals were accepted only through the complete QP check. Separable 128 slowed from 23.60 to 26.84 ms in the screen, and Orbit 40 showed no usable win. | Reject. Forming and factoring the range-space system cost too much. |
| 74 | Cholesky rank update and downdate reproduced 100 changing synthetic matrices per size to 3.6e−15 sampled backward error. The native hint additionally checked every supplied Hessian entry against the secant formula. | Reject. The full Orbit trial failed its 10% target after verification and fallback costs. |
| 75 | Fixed-coordinate substitution handled nonzero pins, all-fixed and changing fixed sets up to 100 variables. Lifted duals were recomputed from original stationarity. Tested BQPGABIM and both DTOC controls. | Archive. The all-fixed case sped up 3.75×. The 50% and 75% pinned fixtures and the three captured problems were slower with substitution. |
| 76 | Quadratic backtracking cut BQPGABIM from 279 to 131 function calls. COOLHANS lost verification and CATENARY reached the step limit instead of converging in 67 steps. | Reject for regression. |
| 77 | A first-trial rejection triggered a bounded step only when the first proposed displacement was long. The 36 historical gain/loss cases exposed both recoveries and losses. The full run moved from 814 to 820 verified, with 16 gains and 10 losses. The bound-constrained subset lost one net. | Reject for regression. |
| 78 | Slack reduction and prior dual size determined bounded elastic penalty growth. The 23-case screen retained 22 verified points and the dedicated elastic/restoration/SOC tests passed. | Reject. No extra verified case or convincing elastic-work improvement. |
| 79 | Restoration radius used predicted and actual merit reduction. UBH5 remained unverified under both tested backends. Neither targeted difficult AC case gained an accepted point. | Reject. |
| 80 | An eight-entry feasibility/objective filter tried a step after repeated merit rejections. It added no verified point to the 23-case screen or the Winter12Generous starts. | Reject. |
| 81 | Precomputed numeric slots sped up repeated sparse Cholesky on narrow and wider bands. The high-fill case declined the oversized table. The 51-/101-arc solves had identical steps and no repeatable end-to-end gain in the screen. | Archive the kernel and its memory guard. |
| 82 | AMD-ordered symbolic admission allowed 20/20 checked sparse polish attempts on a 3,300-dimensional system excluded by the old dense upper bound. The tested AC starts had identical verdicts. | Archive. The measured new coverage was on the isolated sparse fixture. |
| 83 | Retaining the symbolic polish factor cut the repeated constant-signature median from 1.02 to 0.38 ms, while refreshing all coefficients and recomputing the factor. No gain appeared in the affected complete AC trial. | Archive. |
| 84 | A precomputed restoration scatter map matched every tested Hessian entry and gradient. On 60 assembly calls at 640 variables, time fell from 503 to 212 ms after 12 ms of setup. End-to-end eight-solve batches gained below the gate. | Archive the map. |
| 85 | Bulk binary64 transport handled arrays and memoryviews with the original iterable fallback. Strided arrays, foreign endian arrays, generators and wrong lengths were exercised. The sequential screen showed sensitivity to host load, including changes in the list control. | Archive. The current evidence does not establish a reliable end-to-end binding win. |
| 86 | Three-colour HVP recovery reproduced the values of the tridiagonal model. The fair control returns entries on the already declared CSR pattern, and was 1.65–3.10× faster than the three-colour path on this model. | Archive the HVP route. The evidence uses an analytic synthetic model and does not establish an application gain. |
| 87 | Private element work ran concurrently at 1/2/4/8 threads, followed by the same ordered assembly and solution checks. No consistent complete-solve win at higher thread counts. | Reject. |
| 88 | Branch-local Newton KKT factors reproduced the monolithic solution on the published fan fixture. The fresh full-QP trial passed for one of six continued activity guesses and zero of six empty guesses. A rejected candidate needs a cold Clarabel solve. | Archive. The complete constrained solve did not inherit the Newton-kernel gain. |
| 89 | Cached immutable analysis improved the 16-job single-worker median from 215 to 197 ms. Cold one-job and four-worker measurements did not improve. Numerical state remained worker-local in the probe. | Archive. The single-entry process cache added complexity for one narrow gain. |
| 90 | Longest-estimated-first ordering reduced the mixed-orbit makespan at 4×2 from 492 to 436 ms on the screen. At 8×1 it increased from 397 to 494 ms. Every run preserved checked results and input order. | Reject the fixture-based policy. |

## Validation

- Release workspace Rust suite and doctests passed with **237 checks**. The final Python extension passed **263 tests**, including the fan QP bridge and the exact zero-Hessian guard. Rustfmt and Clippy ran successfully. Cargo reported its existing runner-dependency warning.
- All seven numerical tutorial smoke runs and the shared job tutorial passed.
- The full S2MPJ sweep of the baseline and the automatic-tiling trial covered 1,036 problems. It verified **814 → 812** under the fixed deadlines. There were **0 false convergence claims**. 812 of 812 common verified solutions had bit-identical primal vectors. The candidate reached the watchdog on DRCAV3LQ, DUAL2. The same-budget single-worker rerun also hit the watchdog on both. The delivered patch requires an explicit request to use tiling.
- The 14-start EPIGRIDS runs verified 5 on the baseline and 2 on the patched source, with 0 and 8 deadline stops. This is a budget-censored result from runs with differing host load.
- The full timing panels used the pre-promotion builds identified in the environment record. The final test runs used the delivered APIs, including the explicit tile flag and the spectral diagonal guard. The combined element/spectral smoke also used the guard.

The core tolerances were those of the repository. The native and SQP5 gates used primal residual ≤1e−7, stationarity and complementarity ≤1e−6 and dual-sign residual ≤1e−8. The independent S2MPJ and AC checks used their original conventions. The log reports budget-limited runs with their resulting residuals. Equivalent tested objective values use the benchmark's existing comparison tolerance. The preserved raw records include endpoints, work counts and exceptions for diagnostic replay. The protected set and the deterministic 185-case holdout manifest are stored alongside the full S2MPJ records.

See `README.md` in the ZIP for the two diffs, environment, reproducible commands and the evidence manifest. The production diff applies directly to the pinned base commit.
