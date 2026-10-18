# Iteration 04: results

Retained: ordinary QP acceptance validates mutable structure once, checks all derived arithmetic and multiplier inputs, and recomputes the same independent KKT residuals. Added malformed/nonfinite multiplier regressions. The public and elastic entry points retain full validation.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-04.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 12.6928 | 10.8306 | 1.1719 |
| kernel | hessian 128 | 14.2901 | 13.5797 | 1.0523 |
| kernel | csr 128 | 0.4951 | 0.6697 | 0.7393 |
| kernel | csr-dense 128 | 5.1734 | 6.4445 | 0.8028 |
| kernel | transpose 128 | 0.5084 | 0.5941 | 0.8558 |
| kernel | kkt 128 | 18.3019 | 15.8725 | 1.1531 |
| kernel | bfgs 128 | 8.0197 | 7.4831 | 1.0717 |
| kernel | cholesky 96 | 3.9269 | 4.0304 | 0.9743 |
| kernel | inverse 96 | 9.7334 | 9.9131 | 0.9819 |
| kernel | qp-dense 64 | 4.1315 | 3.8858 | 1.0632 |
| kernel | qp-pairs 64 | 1.8771 | 1.8590 | 1.0097 |
| kernel | qp-free 128 | 6.8665 | 7.2842 | 0.9427 |
| kernel | qp-box 128 | 7.1822 | 7.5386 | 0.9527 |
| kernel | qp-zero 64 | 2.0736 | 1.8276 | 1.1346 |
| nlp | orbit 20 | 37.0627 | 36.3966 | 1.0183 |
| nlp | orbit 40 | 319.2065 | 294.0156 | 1.0857 |
| nlp | separable 128 | 20.0127 | 20.0214 | 0.9996 |
| nlp | restoration 64 | 0.9568 | 0.9151 | 1.0455 |
