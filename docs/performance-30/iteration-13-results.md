# Iteration 13: results

Retained: exact-curvature BFGS updates copy y directly; Powell damping and augmented repairs retain their original formulas. The strict matrix-expression regression remains bit-identical on its fixtures.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-13.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 12.2759 | 10.7868 | 1.1381 |
| kernel | hessian 128 | 2.9792 | 3.0110 | 0.9895 |
| kernel | csr 128 | 0.8424 | 0.7842 | 1.0743 |
| kernel | csr-dense 128 | 5.0540 | 5.3657 | 0.9419 |
| kernel | transpose 128 | 0.5554 | 0.5542 | 1.0022 |
| kernel | kkt 128 | 13.6530 | 10.8728 | 1.2557 |
| kernel | bfgs 128 | 5.0564 | 4.9594 | 1.0196 |
| kernel | cholesky 96 | 3.7499 | 3.9306 | 0.9540 |
| kernel | inverse 96 | 9.3664 | 9.6949 | 0.9661 |
| kernel | qp-dense 64 | 3.2640 | 3.1818 | 1.0258 |
| kernel | qp-pairs 64 | 1.7880 | 1.7370 | 1.0294 |
| kernel | qp-free 128 | 6.7392 | 7.1741 | 0.9394 |
| kernel | qp-box 128 | 7.1674 | 7.7962 | 0.9193 |
| kernel | qp-zero 64 | 1.9056 | 1.7531 | 1.0870 |
| nlp | orbit 20 | 31.5422 | 30.7269 | 1.0265 |
| nlp | orbit 40 | 283.0843 | 280.6014 | 1.0088 |
| nlp | separable 128 | 17.7181 | 17.9825 | 0.9853 |
| nlp | restoration 64 | 0.8609 | 0.8932 | 0.9637 |
