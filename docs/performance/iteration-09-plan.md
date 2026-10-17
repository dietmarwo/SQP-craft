# Add an explicit block-diagonal damped-BFGS mode

Hypothesis: user-selected curvature blocks reduce QP Hessian storage and numeric
factorization cost from a dense n-by-n block to independent small blocks.
Add hessian_block_size (0 preserves full BFGS), per-block damped updates and a
stable block CSR pattern. Factor disconnected stored Hessian components
separately in the adapter's SPD check; the full-pattern path is unchanged.

This is an optional approximation, not inferred application structure. Keep
full BFGS as default. Test block SPD, exact absence of cross-block entries,
known-optimum separable and restoration models, invalid options, original
full-mode regressions and unchanged KKT tolerances. Measure block sizes on the
orbit fixture too; failed/reference-mismatching runs are recorded as failures.

Final-validation refinement: floating-point block updates can lose definiteness
near stationarity. Fresh Cholesky checks are inexpensive at block dimension.
On a failed block update, consume the existing Hessian-reset budget, switch
once to full BFGS, invalidate the QP template and report
block_hessian_fallbacks. Repeat the original acceptance tests and retain the
initial unsuccessful block experiments in the record.
