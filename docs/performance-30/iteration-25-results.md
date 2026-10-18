# Iteration 25: results

Retained: violation scans use a fresh ascending bounded-source list rather than reconsidering free sources after every working-set change. Added a same-backend regression that mutates variable and row bound kinds and independently checks every solution.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-25.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.1614 | 9.8580 | 1.0308 |
| kernel | hessian 128 | 3.5185 | 3.0026 | 1.1718 |
| kernel | csr 128 | 0.7627 | 0.9841 | 0.7750 |
| kernel | csr-dense 128 | 4.6569 | 4.6794 | 0.9952 |
| kernel | transpose 128 | 0.6428 | 0.5861 | 1.0969 |
| kernel | kkt 128 | 10.8975 | 11.2765 | 0.9664 |
| kernel | bfgs 128 | 5.0806 | 5.0397 | 1.0081 |
| kernel | cholesky 96 | 3.8240 | 3.9273 | 0.9737 |
| kernel | inverse 96 | 9.6315 | 9.9023 | 0.9727 |
| kernel | qp-dense 64 | 3.1961 | 3.3275 | 0.9605 |
| kernel | qp-pairs 64 | 0.6848 | 0.7008 | 0.9772 |
| kernel | qp-free 128 | 1.2629 | 1.2526 | 1.0083 |
| kernel | qp-box 128 | 1.3582 | 1.3074 | 1.0389 |
| kernel | qp-zero 64 | 0.7603 | 0.7177 | 1.0594 |
| kernel | qp-free-rows 128 | 2.0837 | 2.0496 | 1.0167 |
| kernel | qp-fixed 64 | 0.6488 | 0.7449 | 0.8710 |
| kernel | qp-dense-normals 64 | 4.3725 | 4.3927 | 0.9954 |
| nlp | orbit 20 | 29.9343 | 28.7112 | 1.0426 |
| nlp | orbit 40 | 255.2283 | 253.8918 | 1.0053 |
| nlp | separable 128 | 16.4614 | 15.6720 | 1.0504 |
| nlp | restoration 64 | 0.6996 | 0.7133 | 0.9808 |
