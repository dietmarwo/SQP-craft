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
