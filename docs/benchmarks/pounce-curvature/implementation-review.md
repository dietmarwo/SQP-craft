# SQP-craft / POUNCE curvature review, v2

26 November 2026 · SQP-craft 0.20.0 · POUNCE 0.12.0 · CUTEst problems via S2MPJ

> The delivered log of the pilot's second revision, kept as delivered apart
> from this note. Paths such as `evidence/…` refer to the delivery archive
> `SQP-craft-POUNCE-review-v2-patch.zip` (SHA-256
> `a0e005a94619283f7afdcbbcc09c894944ef0138ffd3ec0851e918c1c823601e`).
> Its counts of points "verified with supplied duals" (14 for both POUNCE
> arms, 3 for damped BFGS) and the ratio of 0.750 on 14 pairs predate the
> rule for fixed variables that the review of the delivery added. The
> [README](README.md#results) has the current figures and the reason.

**Native Clarabel exact curvature improved from 18/20 to 19/20 independently verified first-order solutions on the frozen panel.** CATENARY moved from a failed QP to convergence with the solver’s own multipliers. The other 19 native exact runs reproduced their previous candidates, statuses, callback counts and verified residuals. The original three-arm comparison still gives 20/20 for SQP-craft default BFGS, 19/20 for POUNCE exact, and 20/20 for POUNCE L-BFGS.

The ZIP provides a cumulative patch for the original base and an incremental patch for a checkout with v1 applied. Both application routes produced Git tree `8abf296a86a8fbc74b58805ca510356e7f18c7ef`. The ZIP’s README gives the application and replay commands.

## Findings and changes

| Review finding | Change in v2 | Evidence |
|---|---|---|
| The independent KKT checker reconstructed the Jacobian from the sparsity pattern given to the solver. A missing derivative could produce a false success. | Read the full Jacobian directly from the original S2MPJ model at the candidate. Validate row and variable multiplier dimensions separately. | A smooth cubic row has a missing derivative of 0.75 at the candidate and zero at all three pattern probes. V1 accepted the point. V2 rejects it. The supplied multipliers have stationarity residual 0.75. |
| The weighted Hessian callback formed a unit-weight Lagrangian Hessian and subtracted the objective Hessian when the objective weight was zero. Small constraint curvature could disappear in floating-point subtraction. | For nonunit objective weights, form the requested weighted terms directly. At zero weight, omit the objective evaluation. Keep the existing fast path for unit weight. | With an objective Hessian of 2×10²⁰ I, the expected lower triangle is [0, 1, 2]. V1 returned [0, 1, 0]. V2 returned [0, 1, 2]. |
| Collapsing POUNCE’s two box-dual arrays into a signed vector could hide equal multipliers on inactive lower and upper bounds. | Check bound presence, signs and complementarity on each supplied side before crediting a solver certificate. Preserve the independent verdict when a recovered certificate passes. | A synthetic point with cancelling unit duals on two inactive sides previously looked stationary and complementary. V2 measures complementarity 1, rejects the supplied duals, and independently certifies the point with recovered duals. |
| A zero exact Hessian with nonzero linear objective received only a 10⁻⁸ numerical shift. CATENARY’s unrestricted descent direction produced a huge tangent step. | After the backend rejects this specific matrix, use at least the existing `initial_hessian_scale` in scaled QP coordinates. The usual retry, line search, budgets, and independent QP and NLP acceptance checks still apply. | CATENARY converged in 20 iterations with 21 function, 21 derivative and 21 exact-Hessian evaluations. The analytic circle test checks the retry matrix, a user-specified scale and the zero-gradient case. |
| The tables hid repeated model evaluations and the report command could succeed with failed evidence. | Report actual grouped attempts alongside distinct points and show how many accepted points used supplied or recovered duals. Add `--recheck` to replay saved candidates and `--check` to enforce the correctness checks in the manual workflow. Check configuration headers against their digest and the frozen selection. | BRYBND exact took 605 grouped function and 500 grouped derivative evaluations across 307 and 203 distinct points. Six corrupted inputs each made the report command exit with code 1. |

The archive contains runnable reproductions for the Jacobian and weighted-Hessian findings under `evidence/reproduce-callbacks.py`, with output from both versions. The new regression tests also cover missing objectives, altered residuals, multiplier conversion and the native CATENARY solve.

## Measured outcomes

Each row below has its own denominator. The four native exact control cases also belong to the supplemental 20-case native exact panel. “Verified” means the fresh first-order KKT check passed. “Strict” means the solver also claimed convergence. A recovered multiplier vector can certify the point even when the solver’s supplied duals fail.

| Run | Verified | Strict and verified | Verified with supplied duals | False strict claims |
|---|---:|---:|---:|---:|
| SQP-craft default BFGS | 20/20 | 20 | 20 | 0 |
| POUNCE exact | 19/20 | 18 | 14 | 0 |
| POUNCE L-BFGS | 20/20 | 20 | 14 | 0 |
| POUNCE damped BFGS control | 4/4 | 4 | 3 | 0 |
| SQP-craft Clarabel BFGS control | 4/4 | 4 | 4 | 0 |
| SQP-craft Clarabel exact control | 4/4 | 4 | 4 | 0 |
| Supplemental SQP-craft Clarabel exact, v1 | 18/20 | 18 | 18 | 0 |
| Supplemental SQP-craft Clarabel exact, v2 | 19/20 | 19 | 19 | 0 |

On **CATENARY**, the v1 native solution attempt ended during restoration with primal violation 1.504×10²². V2 returned objective −8184.810369269712, primal violation 1.125×10⁻¹¹, stationarity 3.638×10⁻⁸ and complementarity 1.115×10⁻⁸. The separately installed POUNCE exact comparator still failed this case.

On the 19 comparable, independently verified POUNCE exact/L-BFGS pairs, the median exact-to-L-BFGS ratio was **0.426** for both distinct first-order points and actual grouped first-order attempts. Exact had a median of seven Hessian calls. Restricting the comparison to the 14 pairs where both solvers supplied valid duals gave an actual-attempt ratio of **0.750**. BRYBND’s POUNCE exact candidate passed KKT at the 500-iteration limit. The report retains that distinction.

All 72 primary and control records reproduced their v1 candidates, statuses, evaluation counts and KKT verdicts. Both POUNCE release fixtures passed. Every returned candidate’s reported objective agreed with fresh evaluation in the delivered primary, control and supplemental native panels.

## Why the broader shift change was dropped

I also tried a separate shift for each Hessian diagonal, using that row’s absolute off-diagonal sum to obtain diagonal dominance. It passed first-order verification on 19/20 cases but gave mixed results. These exploratory rows are in `evidence/local-native-exact.jsonl`.

| Case | Delivered uniform-shift policy, F+D | Exploratory row-wise policy, F+D | Outcome |
|---|---:|---:|---|
| HS71 | 114 | 100 | Shifts fell from 57 to 50. Both points verified. |
| DTOC2 | 28 | 55 | Shifts rose from 5 to 21. Both points verified. |
| POWELLBS | 28 | 28 | The delivered version claimed convergence with supplied duals. The exploratory version stalled and needed recovered duals to verify. |
| HS38 | 1,002 | 1,002 | Both exhausted 500 iterations. The delivered version’s stationarity residual was 0.00908. |

The delivered change addresses the flat-Hessian failure. HS71 and HS38 remain useful cases for studying full-space indefinite curvature. The pilot gives no reason to add a public callback yet. It covers at most 58 variables, and the exact adapters emit a full lower triangle. End-to-end memory scaling and the 1,036-problem campaign remain open.

## Verification and reproduction

| Check | Result |
|---|---|
| Git application | Cumulative patch applied to clean base `a87fde4ef8904198f8ba959d047348f3934a7d25`. Incremental patch applied over the delivered v1 patch. Both trees matched. |
| Clean application build | Built a new locked release wheel from the cumulative patch. Its complete Python suite passed, **246 tests**. The environment without POUNCE passed 28 focused tests and skipped 14 optional tests. |
| Rust workspace | Locked workspace tests passed, **236 passed**, one pre-existing census test ignored. Formatting and locked Clippy for all workspace targets passed. |
| Original-model replay | `evidence/audit.py` rechecked **190 saved records**. These are 92 v1 records, 92 v2 records, and two independent three-case repeats. The serial repeats covered CATENARY, HS71 and HS38, including one repeat from the cleanly rebuilt wheel. |
| Report reproduction | From the applied checkout, the strict report replay reproduced the delivered tables byte for byte. The six negative replays covered a failed fixture, false strict success, altered raw dual, edited configuration, changed start and changed objective. |
| Runtime and limits | Linux x86-64, CPython 3.12.14, NumPy 2.5.5, SciPy 1.18.1, one numerical thread per worker, two primary workers. Each arm had a 60-second solve budget, a 90-second watchdog and a 500-iteration setting. Native SQP-craft also had its 5,000-call cap. |
| Source and comparator pins | S2MPJ `fea6a70048eaad28b13a08703ddbfdbf65cd9c30`. POUNCE 0.12.0 wheel SHA-256 `fb4e3040d6bb2e90f55640f35733899cff339d8412352be6e1c6e404990613ef`. Benchmark imports use the separate EPL-2.0 wheel. |

The manual GitHub workflow was exercised locally. A hosted run, large-machine campaign and timing comparison were outside this review. Source, build, environment, candidate and callback details for the measured runs accompany the patch.
