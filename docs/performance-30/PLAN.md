# Thirty further SQP-craft performance iterations

Base: https://github.com/dietmarwo/SQP-craft at `4d776bbd89f187fe176be805ab91fd64305a8c44`, version 0.14.0. The existing ten-iteration series is preserved. This new series has checkpoints 01–30 and targets 0.15.0.

The contract is first-order numerical convergence under unchanged original-unit feasibility, stationarity, complementarity and dual-sign checks. Preserve callback definitions, starts, defaults and budgets. Keep both QP backends. No removal of positive-definiteness gates, accepted residual checks, cancellation or numerical-error guards to obtain a speedup.

Before each change write a separate plan identifying the hot path, intended work reduction, correctness risks and validation. Build/test the change, then measure it serially against the preceding retained checkpoint with identical inputs. Keep raw timings and work explanations; revise/revert harmful experiments. Tiny wall-time samples are not a monotonic performance guarantee. Retained checkpoints must reduce meaningful work or show a measured target gain without an unexplained correctness/robustness regression. Final paired comparisons against the immutable original build decide end-to-end claims.

## Measurement and validation

- Existing native `performance` example, unchanged callbacks: orbit 20/40, separable 128, restoration 64, both Clarabel and active-set; cold/reuse recorded separately. Freeze held-out sizes before tuning: orbit 30/48, separable 96/192, restoration 48/96.
- Add a fixed-work microbenchmark of the actual production kernels and planted QPs (dense, paired rows, unconstrained, diagonal box, zero-gradient constrained). No benchmark-only numerical shortcut enters production.
- Warm up, rotate/reverse paired processes, record all samples, medians, binary hashes and host metadata. No build or stress process runs concurrently with timing. A failed solve is never a speedup.
- Per iteration: affected Rust unit/integration tests and the native correctness-asserting benchmark fixtures. Numerical or cache changes also receive adversarial/invalidation fixtures. Full workspace checks at milestones and final; Python checks against the saved baseline; independent random/linear/MPCC/scaled families. Preserve known failure outcomes and do not hide new failures.
- Keep source checkpoints and a reproducible ledger for all 30 iterations. Final delivery: source, patch, individual plans/results, raw data, validation logs, limitations and reproduction commands.

## Candidate sequence (adapt to measurements)

1. Linear-time sorted-CSR index range checks.
2. Constant-time-per-row Hessian lower-triangle validation after CSR validation.
3. Avoid duplicate bound validation in independently checked KKT arithmetic.
4. Reuse validated QP structure within its independent residual check.
5. Row-local symmetric Hessian products.
6. Contiguous dense-lower Hessian product specialization.
7. Contiguous dense CSR row products.
8. Singleton CSR row products.
9. Finite-data zero-multiplier transpose products inside KKT checking.
10. Fuse BFGS finite checks with triangle construction.
11. Use contiguous BFGS update slices without changing scalar formulas.
12. Early negligible-step detection in BFGS.
13. Copy exact-curvature gradient differences without damping arithmetic.
14. Clear only the factor triangle that the active-set backend reads.
15. Overwrite fully stored Hessians without redundant zeroing.
16. Avoid norms for irrelevant free rows.
17. Specialize transformed singleton constraint normals.
18. Stop dependent warm-set insertions once the basis is full.
19. Restrict Householder updates to nonzero support.
20. Build diagonal inverse factors directly.
21. Skip zero coefficients when accumulating KKT primal directions.
22. Avoid computing activity error magnitudes where only activity is needed.
23. Borrow the unmodified gradient in nonelastic polishing.
24. Reserve retained working-set storage once.
25. Restrict violation scans to bounded sources.
26. Reuse the equality source list within an attempt/retry.
27. Checked unconstrained-QP triangular-solve path.
28. Checked diagonal box-QP path.
29. Reuse Cholesky factors only for bit-identical Hessian data/structure.
30. Reuse the initial inverse factor only when its Hessian cache is valid.

These are engineering hypotheses, not promised per-instance wins. If a candidate fails, the iteration records that attempt and adopts a measured replacement before advancing. Larger algorithmic approximations and changes to defaults require separate evidence and are outside this series.

## Completed iteration ledger

All 30 iterations are implemented, tested and retained as local source tags. Each linked plan was written before its implementation; its result records the measured work, tests and counterexamples. The final 0.15.0 release adds the audited two-entry CSR refinement and integration/documentation fixes after checkpoint 30.

| Iteration | Implementation plan | Tests and measurements |
|---:|---|---|
| 01 | [Use the sorted CSR row endpoint for range checks](iteration-01-plan.md) | [Result](iteration-01-results.md) |
| 02 | [Check the Hessian triangle through sorted row endpoints](iteration-02-plan.md) | [Result](iteration-02-results.md) |
| 03 | [Reuse already validated bound structure in KKT checks](iteration-03-plan.md) | [Result](iteration-03-results.md) |
| 04 | [Share checked KKT arithmetic after QP validation](iteration-04-plan.md) | [Result](iteration-04-results.md) |
| 05 | [Accumulate symmetric Hessian rows locally](iteration-05-plan.md) | [Result](iteration-05-results.md) |
| 06 | [Use contiguous arithmetic for dense lower Hessian rows](iteration-06-plan.md) | [Result](iteration-06-results.md) |
| 07 | [Use contiguous products for full CSR rows](iteration-07-plan.md) | [Result](iteration-07-results.md) |
| 08 | [Specialize singleton CSR row products](iteration-08-plan.md) | [Result](iteration-08-results.md) |
| 09 | [Skip zero multiplier rows after finite-data validation](iteration-09-plan.md) | [Result](iteration-09-results.md) |
| 10 | [Fuse finite BFGS validation with triangle construction](iteration-10-plan.md) | [Result](iteration-10-results.md) |
| 11 | [Separate contiguous BFGS triangle computation from mirroring](iteration-11-plan.md) | [Result](iteration-11-results.md) |
| 12 | [Stop negligible-step scans once a material component is found](iteration-12-plan.md) | [Result](iteration-12-results.md) |
| 13 | [Copy exact-curvature differences without damping arithmetic](iteration-13-plan.md) | [Result](iteration-13-results.md) |
| 14 | [Clear only the factor triangle that is read](iteration-14-plan.md) | [Result](iteration-14-results.md) |
| 15 | [Avoid clearing completely stored Hessian triangles](iteration-15-plan.md) | [Result](iteration-15-results.md) |
| 16 | [Avoid unused row norms and extend working-set coverage](iteration-16-plan.md) | [Result](iteration-16-results.md) |
| 17 | [Project singleton constraint normals directly](iteration-17-plan.md) | [Result](iteration-17-results.md) |
| 18 | [Skip dependent insertions after the basis is full](iteration-18-plan.md) | [Result](iteration-18-results.md) |
| 19 | [Restrict Householder matrix updates to exact nonzero support](iteration-19-plan.md) | [Result](iteration-19-results.md) |
| 20 | [Build exact diagonal inverse factors directly](iteration-20-plan.md) | [Result](iteration-20-results.md) |
| 21 | [Skip zero KKT primal direction coefficients](iteration-21-plan.md) | [Result](iteration-21-results.md) |
| 22 | [Compute only the constraint activity when its error scale is unused](iteration-22-plan.md) | [Result](iteration-22-results.md) |
| 23 | [Borrow the original gradient during nonelastic polishing](iteration-23-plan.md) | [Result](iteration-23-results.md) |
| 24 | [Reserve the retained working set once per dimension](iteration-24-plan.md) | [Result](iteration-24-results.md) |
| 25 | [Scan only bounded constraint sources for violations](iteration-25-plan.md) | [Result](iteration-25-results.md) |
| 26 | [Reuse a fresh equality-source list across attempts](iteration-26-plan.md) | [Result](iteration-26-results.md) |
| 27 | [Solve fully unconstrained QPs from the checked Cholesky factor](iteration-27-plan.md) | [Result](iteration-27-results.md) |
| 28 | [Check diagonal box minimizers before general working-set search](iteration-28-plan.md) | [Result](iteration-28-results.md) |
| 29 | [Reuse Cholesky factors only for bit-identical Hessians](iteration-29-plan.md) | [Result](iteration-29-results.md) |
| 30 | [Cache the unrotated inverse only after confirmed Hessian reuse](iteration-30-plan.md) | [Result](iteration-30-results.md) |

Final paired results: the active-set four-fixture geometric speedup is 1.247×, with orbit-40 falling from 282.94 to 253.08 ms. All six declared held-out active-set sizes improve. Default Clarabel results are mixed; individual noisy/tiny regressions remain visible. There is no claim of monotonically faster wall time at every checkpoint.

Validation: 173 regular Rust tests, 135 Python tests, 17,372 independent baseline/final pairs (34,744 solver calls), unchanged audited acceptance/status/outer iterations and a matching 1000-case harder QP outcome census. See [RESULTS.md](RESULTS.md) for the complete limitations, [TABLES.md](TABLES.md) for exact timing tables, [VALIDITY.md](VALIDITY.md) for numerical arguments and [REPRODUCE.md](REPRODUCE.md) for commands.
