# Iteration 20: results

Retained: exactly diagonal Cholesky factors bypass cubic zero-work inverse solves; any nonzero coupling retains the full algorithm. Added analytic diagonal, empty and tiny-coupling checks. No numerical nonzero is dropped.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-20.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.4352 | 10.6736 | 0.9777 |
| kernel | hessian 128 | 2.7061 | 2.8506 | 0.9493 |
| kernel | csr 128 | 0.8335 | 0.8700 | 0.9580 |
| kernel | csr-dense 128 | 4.6360 | 4.7256 | 0.9811 |
| kernel | transpose 128 | 0.5363 | 0.5502 | 0.9747 |
| kernel | kkt 128 | 10.8470 | 10.8072 | 1.0037 |
| kernel | bfgs 128 | 5.1135 | 5.1315 | 0.9965 |
| kernel | cholesky 96 | 3.8214 | 4.1581 | 0.9190 |
| kernel | inverse 96 | 9.3273 | 9.4244 | 0.9897 |
| kernel | qp-dense 64 | 3.2297 | 3.2980 | 0.9793 |
| kernel | qp-pairs 64 | 1.8600 | 0.7163 | 2.5968 |
| kernel | qp-free 128 | 6.7833 | 1.3534 | 5.0119 |
| kernel | qp-box 128 | 6.9053 | 1.5161 | 4.5545 |
| kernel | qp-zero 64 | 1.7307 | 0.6927 | 2.4985 |
| kernel | qp-free-rows 128 | 8.0664 | 2.2542 | 3.5783 |
| kernel | qp-fixed 64 | 1.8743 | 0.8270 | 2.2663 |
| kernel | qp-dense-normals 64 | 5.5072 | 4.0061 | 1.3747 |
| nlp | orbit 20 | 30.7634 | 29.7217 | 1.0350 |
| nlp | orbit 40 | 267.2423 | 269.0931 | 0.9931 |
| nlp | separable 128 | 16.9778 | 16.6958 | 1.0169 |
| nlp | restoration 64 | 0.8219 | 0.7774 | 1.0571 |
