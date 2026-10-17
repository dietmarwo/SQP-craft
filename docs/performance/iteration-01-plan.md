# Compute one BFGS triangle and mirror it

Hypothesis: the symmetric rank-two update computes the same off-diagonal
expression twice. Compute each value once and mirror it, preserving the exact
matrix and the Cholesky safeguard. This halves update-expression evaluations.

Implement only this kernel change. Validate bit equality against the existing
matrix-expression reference, damping/overflow tests, the full release suite,
and the four fixed-callback benchmarks. Report BFGS phase and total runtime;
small overall differences may remain below host noise.
