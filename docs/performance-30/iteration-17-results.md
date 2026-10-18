# Iteration 17: results

Retained: singleton row projection is one direct strided gather per transformed component. Coefficient/orientation arithmetic keeps its original order, and general sparse-row projection remains unchanged. Random sparse and elastic QP checks pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-17.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.2185 | 10.2056 | 1.0013 |
| kernel | hessian 128 | 2.7110 | 2.7872 | 0.9727 |
| kernel | csr 128 | 0.9600 | 0.8099 | 1.1854 |
| kernel | csr-dense 128 | 5.1860 | 4.9140 | 1.0554 |
| kernel | transpose 128 | 0.5297 | 0.5677 | 0.9331 |
| kernel | kkt 128 | 10.6569 | 11.2192 | 0.9499 |
| kernel | bfgs 128 | 5.3083 | 5.2141 | 1.0181 |
| kernel | cholesky 96 | 3.9451 | 3.6689 | 1.0753 |
| kernel | inverse 96 | 10.5881 | 9.6366 | 1.0987 |
| kernel | qp-dense 64 | 3.6615 | 3.2579 | 1.1239 |
| kernel | qp-pairs 64 | 1.7913 | 1.7922 | 0.9995 |
| kernel | qp-free 128 | 6.9415 | 6.7179 | 1.0333 |
| kernel | qp-box 128 | 7.8063 | 8.0841 | 0.9656 |
| kernel | qp-zero 64 | 1.6944 | 1.6839 | 1.0063 |
| kernel | qp-free-rows 128 | 7.3894 | 7.3497 | 1.0054 |
| kernel | qp-fixed 64 | 1.9041 | 1.7418 | 1.0932 |
| kernel | qp-dense-normals 64 | 4.9829 | 5.0378 | 0.9891 |
| nlp | orbit 20 | 29.4285 | 29.9262 | 0.9834 |
| nlp | orbit 40 | 251.0765 | 282.0018 | 0.8903 |
| nlp | separable 128 | 16.5876 | 17.5790 | 0.9436 |
| nlp | restoration 64 | 0.8921 | 0.8865 | 1.0063 |
