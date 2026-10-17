# Certify strictly diagonally dominant Hessians in linear storage time

Hypothesis: identity, diagonal restoration and many early Hessians need no
cubic Cholesky to establish positive definiteness. For a symmetric matrix,
positive diagonal entries strictly exceeding each row's sum of absolute
off-diagonals imply positive eigenvalues (Gershgorin). Accumulate those
nonnegative sums upward by one binary64 ulp per nonzero addition; overflow
makes the sufficient test fail. All other matrices retain the existing full
Cholesky gate. No approximate drop threshold or weakened QP residual check.

Test positive, indefinite, semidefinite, non-dominant positive-definite and
near-boundary examples; run full release tests and unchanged benchmarks.
