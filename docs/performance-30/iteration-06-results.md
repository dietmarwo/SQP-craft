# Iteration 06: results

Retained: full lower Hessian rows use contiguous slices with no index gathers. Sparse and block patterns retain the general path. The fixed arithmetic order, model validation and independent residual thresholds are preserved.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-06.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.3177 | 11.9747 | 0.8616 |
| kernel | hessian 128 | 3.2951 | 2.8109 | 1.1722 |
| kernel | csr 128 | 0.4997 | 0.4762 | 1.0493 |
| kernel | csr-dense 128 | 5.5846 | 4.8013 | 1.1631 |
| kernel | transpose 128 | 0.5127 | 0.5196 | 0.9868 |
| kernel | kkt 128 | 15.4971 | 16.1170 | 0.9615 |
| kernel | bfgs 128 | 6.8891 | 7.1180 | 0.9679 |
| kernel | cholesky 96 | 3.5573 | 3.6084 | 0.9858 |
| kernel | inverse 96 | 9.0665 | 9.3066 | 0.9742 |
| kernel | qp-dense 64 | 3.1670 | 3.1826 | 0.9951 |
| kernel | qp-pairs 64 | 1.6828 | 1.7168 | 0.9802 |
| kernel | qp-free 128 | 6.6377 | 6.5306 | 1.0164 |
| kernel | qp-box 128 | 6.5823 | 6.6129 | 0.9954 |
| kernel | qp-zero 64 | 1.7162 | 1.6823 | 1.0202 |
| nlp | orbit 20 | 30.1400 | 30.4241 | 0.9907 |
| nlp | orbit 40 | 257.1237 | 267.0506 | 0.9628 |
| nlp | separable 128 | 16.9763 | 17.8493 | 0.9511 |
| nlp | restoration 64 | 0.8572 | 0.9174 | 0.9344 |
