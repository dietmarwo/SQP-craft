# Maintain the BFGS Cholesky factor with rank updates

Hypothesis: refactorizing the dense Hessian after every accepted step costs
O(n^3). Apply the positive rank-one term before the downdate, using nalgebra's
Cholesky updates, and rebuild every 32 updates or after an invalid factor.
The BFGS matrix expression stays identical; the QP adapter retains its separate
positive-definiteness gate on every model. Reset the factor whenever the driver
resets B, including restoration.

Validate repeated factor residuals against the independently formed BFGS
matrix, the existing bit-equality reference, damping/overflow, full release
regressions, and fixed-callback benchmarks. Reject any loss of original-unit
KKT acceptance. This optimizes a safeguard, not the SQP convergence test.

Final-validation refinement: a small random problem exposed cancellation
between the explicitly stored rank-two terms even while the updated factor
remained finite. Preserve fresh Cholesky checks for dimensions <=16 (where
cubic work is inexpensive), and for each explicit block. The captured seed-4,
case-86 input is a permanent Python regression. Larger full Hessians retain
periodic/fallback factor rebuilds and the adapter's independent SPD gate.
