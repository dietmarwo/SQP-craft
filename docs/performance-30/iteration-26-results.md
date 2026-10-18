# Iteration 26: results

Retained: each model load builds ordered bounded/equality lists in one pass. Initial and retry basis construction reuse the fresh equality list, and the mutation regression now checks both lists. Equality ordering and hard/soft-row semantics are unchanged.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-26.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 9.7938 | 9.6885 | 1.0109 |
| kernel | hessian 128 | 2.6776 | 2.9977 | 0.8932 |
| kernel | csr 128 | 0.7625 | 0.7851 | 0.9713 |
| kernel | csr-dense 128 | 4.5808 | 4.6622 | 0.9825 |
| kernel | transpose 128 | 0.5444 | 0.5495 | 0.9908 |
| kernel | kkt 128 | 10.6237 | 10.9093 | 0.9738 |
| kernel | bfgs 128 | 4.8951 | 5.0416 | 0.9709 |
| kernel | cholesky 96 | 3.6173 | 3.6973 | 0.9784 |
| kernel | inverse 96 | 9.2852 | 9.0671 | 1.0240 |
| kernel | qp-dense 64 | 3.2269 | 3.1315 | 1.0305 |
| kernel | qp-pairs 64 | 0.6895 | 0.7100 | 0.9711 |
| kernel | qp-free 128 | 1.2485 | 1.2127 | 1.0295 |
| kernel | qp-box 128 | 1.3483 | 1.3367 | 1.0087 |
| kernel | qp-zero 64 | 0.6506 | 0.6463 | 1.0067 |
| kernel | qp-free-rows 128 | 2.0626 | 2.1093 | 0.9778 |
| kernel | qp-fixed 64 | 0.6454 | 0.6668 | 0.9679 |
| kernel | qp-dense-normals 64 | 4.1731 | 4.2485 | 0.9823 |
| nlp | orbit 20 | 28.7352 | 29.1031 | 0.9874 |
| nlp | orbit 40 | 244.3170 | 252.5403 | 0.9674 |
| nlp | separable 128 | 15.9078 | 15.8820 | 1.0016 |
| nlp | restoration 64 | 0.7600 | 0.7671 | 0.9908 |
