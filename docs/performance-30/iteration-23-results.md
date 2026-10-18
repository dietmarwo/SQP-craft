# Iteration 23: results

Retained: nonelastic KKT solves and refinement borrow the immutable input gradient. Elastic gradients are still rebuilt for every polish, including changed weights and elastic-side transitions. Ordinary and elastic stress QPs pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-23.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 9.9507 | 11.7468 | 0.8471 |
| kernel | hessian 128 | 2.7736 | 2.8296 | 0.9802 |
| kernel | csr 128 | 0.8092 | 0.7601 | 1.0645 |
| kernel | csr-dense 128 | 4.7542 | 4.8457 | 0.9811 |
| kernel | transpose 128 | 0.5011 | 0.5518 | 0.9081 |
| kernel | kkt 128 | 10.8939 | 12.7595 | 0.8538 |
| kernel | bfgs 128 | 4.8223 | 4.9233 | 0.9795 |
| kernel | cholesky 96 | 3.6108 | 3.9391 | 0.9166 |
| kernel | inverse 96 | 9.5496 | 9.3764 | 1.0185 |
| kernel | qp-dense 64 | 3.1751 | 3.2410 | 0.9796 |
| kernel | qp-pairs 64 | 0.6965 | 0.7042 | 0.9890 |
| kernel | qp-free 128 | 1.2115 | 1.2493 | 0.9697 |
| kernel | qp-box 128 | 1.3333 | 1.4765 | 0.9030 |
| kernel | qp-zero 64 | 0.6526 | 0.6420 | 1.0165 |
| kernel | qp-free-rows 128 | 2.1083 | 2.2482 | 0.9377 |
| kernel | qp-fixed 64 | 0.7926 | 0.6367 | 1.2449 |
| kernel | qp-dense-normals 64 | 4.5710 | 4.8054 | 0.9512 |
| nlp | orbit 20 | 31.1914 | 30.1155 | 1.0357 |
| nlp | orbit 40 | 246.5364 | 244.8346 | 1.0070 |
| nlp | separable 128 | 17.2656 | 16.9792 | 1.0169 |
| nlp | restoration 64 | 0.7888 | 0.7402 | 1.0657 |
