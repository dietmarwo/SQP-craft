# Reuse independent KKT-check buffers

Hypothesis: each QP is checked by both backend and driver, allocating Hessian
products, row activities and transpose products every time. Add reusable
workspace APIs to the independent checker and give the backend and driver
separate workspaces. Keep both checks and every validation/finite-input rule.

Test allocating/workspace residual equality, dimension changes and malformed
multipliers; run all release tests and the unchanged benchmark suite. This
removes allocations without trusting a backend's convergence flag.
