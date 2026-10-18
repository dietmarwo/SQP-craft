# Iteration 11: results

Retained: lower-triangle BFGS arithmetic is contiguous; mirroring happens after the finite gate. Scalar formulas and exact matrix-expression outputs are unchanged, and a rejected update still leaves B untouched.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-11.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 11.9924 | 9.7685 | 1.2277 |
| kernel | hessian 128 | 2.7548 | 2.8847 | 0.9550 |
| kernel | csr 128 | 0.7606 | 0.7817 | 0.9730 |
| kernel | csr-dense 128 | 4.5121 | 4.5702 | 0.9873 |
| kernel | transpose 128 | 0.5384 | 0.5330 | 1.0102 |
| kernel | kkt 128 | 12.5191 | 10.6318 | 1.1775 |
| kernel | bfgs 128 | 4.0688 | 4.9927 | 0.8149 |
| kernel | cholesky 96 | 4.1337 | 3.6083 | 1.1456 |
| kernel | inverse 96 | 9.2509 | 9.2817 | 0.9967 |
| kernel | qp-dense 64 | 3.3742 | 3.1584 | 1.0683 |
| kernel | qp-pairs 64 | 1.8426 | 1.7147 | 1.0746 |
| kernel | qp-free 128 | 6.5119 | 6.5592 | 0.9928 |
| kernel | qp-box 128 | 6.7323 | 6.6599 | 1.0109 |
| kernel | qp-zero 64 | 1.6964 | 1.7294 | 0.9809 |
| nlp | orbit 20 | 30.5778 | 29.2297 | 1.0461 |
| nlp | orbit 40 | 256.1934 | 247.5877 | 1.0348 |
| nlp | separable 128 | 17.0011 | 16.1425 | 1.0532 |
| nlp | restoration 64 | 0.9459 | 0.8237 | 1.1482 |
