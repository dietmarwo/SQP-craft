# Preserve sparsity in restoration QPs

Hypothesis: Gauss–Newton restoration computes both symmetric triangles, copies
ordinary constraint rows only to discard them, and stores explicit zero
curvature. Compute the lower triangle, construct the unconstrained-row model
directly and omit exact zeros (no numerical dropping threshold). Compute the
squared-violation merit without a temporary residual vector.

Validate the analytic diagonal least-squares model, overflow and existing
restoration regressions, all release tests and the unchanged four benchmarks.
This preserves the exact QP mathematics but can change Clarabel ordering and
iteration trajectories. Keep original-unit acceptance unchanged.
