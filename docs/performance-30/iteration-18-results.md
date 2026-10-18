# Iteration 18: results

Retained: a saturated n-column working set skips projections that cannot be independent. Dependent inconsistent rows are still checked by the ordinary violation/infeasibility path; their rows are never declared satisfied by this shortcut.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-18.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.4102 | 12.2771 | 0.8479 |
| kernel | hessian 128 | 2.9596 | 3.1748 | 0.9322 |
| kernel | csr 128 | 0.7785 | 0.9283 | 0.8387 |
| kernel | csr-dense 128 | 5.2643 | 5.0106 | 1.0506 |
| kernel | transpose 128 | 0.5987 | 0.5892 | 1.0162 |
| kernel | kkt 128 | 11.2481 | 13.7728 | 0.8167 |
| kernel | bfgs 128 | 4.9559 | 5.3223 | 0.9311 |
| kernel | cholesky 96 | 3.7733 | 3.7878 | 0.9962 |
| kernel | inverse 96 | 9.6151 | 9.6254 | 0.9989 |
| kernel | qp-dense 64 | 3.4360 | 3.3429 | 1.0278 |
| kernel | qp-pairs 64 | 1.8760 | 1.7959 | 1.0446 |
| kernel | qp-free 128 | 7.7631 | 7.0027 | 1.1086 |
| kernel | qp-box 128 | 7.6740 | 6.8337 | 1.1230 |
| kernel | qp-zero 64 | 1.8763 | 1.9032 | 0.9859 |
| kernel | qp-free-rows 128 | 7.9776 | 8.1940 | 0.9736 |
| kernel | qp-fixed 64 | 1.8799 | 1.8057 | 1.0411 |
| kernel | qp-dense-normals 64 | 5.3415 | 5.6177 | 0.9508 |
| nlp | orbit 20 | 32.2271 | 30.9056 | 1.0428 |
| nlp | orbit 40 | 260.6799 | 259.9916 | 1.0026 |
| nlp | separable 128 | 19.3106 | 16.4350 | 1.1750 |
| nlp | restoration 64 | 0.8267 | 0.8335 | 0.9919 |
