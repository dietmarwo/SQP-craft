# Iteration 43: Parallel KKT primal reconstruction

Assign disjoint primal-coordinate tiles instead of scattering into one shared x. Within each tile, visit J columns in the same original order and skip exactly zero coefficients, using the same scalar axpy. Keep triangular u solves serial and all working-set choices unchanged. Existing complete c/x/u bit-identity tests cover q=0/17/n, zeros and threshold boundaries. Use n>=256 and a coarse work gate; full release tests plus paired eight-thread/single-thread/batch timing. This avoids atomics and per-worker n-vectors/reductions.
