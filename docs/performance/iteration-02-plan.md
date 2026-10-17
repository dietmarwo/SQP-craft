# Reuse the BFGS workspace

Hypothesis: allocating two dense matrices and several vectors on every update
adds avoidable allocation/initialization work. Keep candidate, Cholesky scratch,
Bs and damped-y buffers for the duration of one solve; borrow s/y via views.
Reuse the factor-check allocation through nalgebra's unpack_dirty API.

Preserve each floating-point update expression and the full Cholesky test.
Validate the existing bit-equality reference, all release tests and the same
four fixed-callback benchmarks. Check that no scratch survives across solves.
