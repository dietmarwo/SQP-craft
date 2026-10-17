# Reuse the SQP step-QP template

Hypothesis: every outer step rebuilds the same Hessian/Jacobian pattern and
allocates shifted bounds, gradient and matrix values. Own one mutable QP model
per solve; refresh all numeric data without changing the model or QP checks.
Recorded QP traces remain independent owned snapshots. Restoration keeps its
separate model because its rows differ.

Test refreshed versus fresh QP conversion, pattern/data allocation stability,
full release tests and fixed-callback benchmarks with unchanged iteration
counts and original-unit residual gates.
