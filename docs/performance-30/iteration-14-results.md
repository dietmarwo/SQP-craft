# Iteration 14: results

Retained: factor initialization writes only the lower triangle consumed by Cholesky/triangular solves. Missing structural entries are still cleared. Existing upper-triangle-NaN, varying-size random QP, warm/cold and indefinite-Hessian tests pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-14.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.4846 | 11.8952 | 0.8814 |
| kernel | hessian 128 | 2.8305 | 2.7572 | 1.0266 |
| kernel | csr 128 | 0.7715 | 0.8072 | 0.9558 |
| kernel | csr-dense 128 | 4.8151 | 4.4216 | 1.0890 |
| kernel | transpose 128 | 0.5184 | 0.5886 | 0.8808 |
| kernel | kkt 128 | 11.0150 | 14.2461 | 0.7732 |
| kernel | bfgs 128 | 4.8244 | 5.1956 | 0.9286 |
| kernel | cholesky 96 | 3.7653 | 4.3245 | 0.8707 |
| kernel | inverse 96 | 9.3243 | 9.6470 | 0.9666 |
| kernel | qp-dense 64 | 3.4968 | 3.2805 | 1.0659 |
| kernel | qp-pairs 64 | 1.7991 | 2.1807 | 0.8250 |
| kernel | qp-free 128 | 6.9388 | 6.9452 | 0.9991 |
| kernel | qp-box 128 | 6.9882 | 6.9387 | 1.0071 |
| kernel | qp-zero 64 | 1.7782 | 2.0035 | 0.8876 |
| nlp | orbit 20 | 31.2208 | 29.8457 | 1.0461 |
| nlp | orbit 40 | 290.5921 | 269.9555 | 1.0764 |
| nlp | separable 128 | 17.9435 | 17.0200 | 1.0543 |
| nlp | restoration 64 | 0.8334 | 0.8543 | 0.9755 |
