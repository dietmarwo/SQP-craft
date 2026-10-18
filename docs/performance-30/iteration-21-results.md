# Iteration 21: results

Retained: exactly zero primal-combination coefficients perform no full-column axpy. Nonzero and nonfinite coefficients are not discarded; all final steps and multipliers still pass the same independent original-data checks.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-21.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.2005 | 10.3778 | 0.9829 |
| kernel | hessian 128 | 2.7131 | 2.7298 | 0.9939 |
| kernel | csr 128 | 0.7800 | 0.7850 | 0.9937 |
| kernel | csr-dense 128 | 4.5566 | 4.7354 | 0.9622 |
| kernel | transpose 128 | 0.5342 | 0.5227 | 1.0221 |
| kernel | kkt 128 | 10.7855 | 12.0518 | 0.8949 |
| kernel | bfgs 128 | 4.9652 | 4.8464 | 1.0245 |
| kernel | cholesky 96 | 3.6677 | 3.6843 | 0.9955 |
| kernel | inverse 96 | 9.4808 | 9.5313 | 0.9947 |
| kernel | qp-dense 64 | 3.3231 | 3.2678 | 1.0169 |
| kernel | qp-pairs 64 | 0.7212 | 0.7099 | 1.0159 |
| kernel | qp-free 128 | 1.3696 | 1.2473 | 1.0981 |
| kernel | qp-box 128 | 1.4082 | 1.4475 | 0.9729 |
| kernel | qp-zero 64 | 0.6993 | 0.6643 | 1.0526 |
| kernel | qp-free-rows 128 | 2.1714 | 2.1306 | 1.0191 |
| kernel | qp-fixed 64 | 0.6867 | 0.6355 | 1.0804 |
| kernel | qp-dense-normals 64 | 4.0093 | 4.3514 | 0.9214 |
| nlp | orbit 20 | 30.1663 | 29.9953 | 1.0057 |
| nlp | orbit 40 | 252.3211 | 256.4427 | 0.9839 |
| nlp | separable 128 | 16.6649 | 16.4549 | 1.0128 |
| nlp | restoration 64 | 0.7252 | 0.7334 | 0.9889 |
