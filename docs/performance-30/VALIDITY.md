# Numerical safety and scope of the performance changes

The production solver remains a numerical first-order SQP pilot. No nonlinear infeasibility, second-order or global optimality certificate is added. Callback formulas, reference objectives, budgets, tolerances, full-BFGS default and default Clarabel backend are unchanged. The active-set backend remains explicitly selected.

## Validation reuse is not unchecked input reuse

CSR rows are still validated for pointer consistency and strict increasing indices. A sorted row's endpoint is its maximum, so valid index-range checks and lower-only Hessian checks need no second full scan. Invalid inputs still produce a concrete offending index. A full sorted row contains every column; a full lower row contains exactly 0..i. Those invariants justify contiguous products.

Public KKT and QP entry points retain validation. Inside one QP acceptance call, mutable model data are validated before products are formed. Step and multiplier dimensions/finiteness and derived arithmetic are checked before the shared internal KKT arithmetic is called. Bounds already validated in that call do not require a second structural validation. The numerical stationarity, primal, complementarity and dual-sign formulas/thresholds are unchanged. Zero-multiplier rows are skipped only inside this finite-data path; the public raw CSR transpose routine retains its arithmetic behavior.

## BFGS and active-set arithmetic

The BFGS triangle is still computed with the same scalar rank-two formulas, and mirrored before committing an update. Finiteness is recorded while computing the triangle; a failed update never swaps it into B. Explicit blocks keep their Cholesky check and fallback. The strict matrix-expression regression and padded-indefinite-Hessian regression remain in force. NaN displacements cannot enter the negligible-step shortcut.

Factor buffers clear all missing lower entries; the unused upper triangle is never read. A fully stored validated lower triangle needs no preceding zero fill. No numerical nonzero is dropped. Diagonal inverse construction requires exact zero off-diagonal entries. Householder support restriction skips only exact zero weights and preserves the full-tail norm reduction. Rounding-error magnitudes remain present in every violation/feasibility decision; activity-only dots are used only where the magnitude was discarded. Source lists are rebuilt from the current bounds on every load, including cold retries and bound-kind changes.

## Checked direct QP proposals

The mandatory Cholesky gate precedes either shortcut. A nonelastic QP without any finite bound has unique minimizer H*x=-g; triangular solves and one original-H residual refinement propose it. For an exactly diagonal positive-definite H, clamping -g_j/H_jj solves the variable-box relaxation, with the corresponding signed variable multipliers. If that relaxation minimizer also satisfies the original row/dual KKT check, it is a valid numerical solution of the full convex QP. Otherwise the general active-set solver runs.

Both proposals pass the unchanged independent original-model acceptance rule and budget/cancellation checks. Failed arithmetic/checks are not an infeasibility or optimality proof. Elastic QPs keep the general path. Accepted closed-form solves report zero working-set changes and return bound activity for subsequent warm starts; exact counter tests were updated to assert this behavior while retaining point, multiplier, callback-count and tolerance checks.

## Exact-key factor and inverse caches

The backend owns a complete copy of the last Hessian's dimensions, CSR pointers/indices and binary64 value bits. No approximate comparison or hash is used. A successful factorization may be reused only on an exact key match. A new key clears validity before any load/factor work; failed factorizations cannot become hits. Invalid input is rejected before touching the cache. The cached conditioning estimate depends only on that same factor.

The optional retained initial inverse is **unrotated** L^-T and its norm, not the final working-set J/R. It depends only on H, so current gradients, Jacobians, bounds and warm sets still receive fresh processing. It is promoted only after confirmed cross-call Hessian reuse, so ordinary changing NLP Hessians do not pay an unused inverse copy. Every Hessian miss invalidates it. A nonfinite inverse/norm is not cached. Reuse copies the initial inverse before normal working-set transformations. Backend clones own independent copies; there is no shared global cache.

Tests cover changed values, equal values under a changed CSR pattern, signed zeros, dimensions, invalid data, indefinite-factor failures, cloned backends, changing row data and cache promotion. Final steps/multipliers always pass original-data acceptance; a cache hit never substitutes for that check. `QpReport.setups` counts new factorization attempts and `updates` counts compatible factor reuses for the active-set backend. The retained key and, on repeated H, one extra dense inverse trade memory for work.
