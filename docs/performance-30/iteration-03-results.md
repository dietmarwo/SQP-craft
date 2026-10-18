# Iteration 03: results

Retained: independent KKT checks no longer validate the same bound structures twice. The crate-private arithmetic path still rejects nonfinite points and overflow; public Bounds::checked_violation retains its full validation contract.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-03.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 12.3226 | 13.3232 | 0.9249 |
| kernel | hessian 128 | 12.6147 | 12.7817 | 0.9869 |
| kernel | csr 128 | 0.7544 | 0.7389 | 1.0210 |
| kernel | csr-dense 128 | 5.1060 | 6.1947 | 0.8243 |
| kernel | transpose 128 | 0.7863 | 0.8820 | 0.8915 |
| kernel | kkt 128 | 18.6463 | 20.3156 | 0.9178 |
| kernel | bfgs 128 | 8.4048 | 8.7574 | 0.9597 |
| kernel | cholesky 96 | 3.7925 | 5.2094 | 0.7280 |
| kernel | inverse 96 | 10.0114 | 9.6395 | 1.0386 |
| kernel | qp-dense 64 | 3.6370 | 3.9156 | 0.9289 |
| kernel | qp-pairs 64 | 1.9362 | 2.0787 | 0.9315 |
| kernel | qp-free 128 | 7.8353 | 7.2764 | 1.0768 |
| kernel | qp-box 128 | 7.6720 | 7.1029 | 1.0801 |
| kernel | qp-zero 64 | 2.2238 | 1.9513 | 1.1397 |
| nlp | orbit 20 | 39.7113 | 39.8033 | 0.9977 |
| nlp | orbit 40 | 309.9727 | 305.0271 | 1.0162 |
| nlp | separable 128 | 21.0687 | 21.5171 | 0.9792 |
| nlp | restoration 64 | 1.0304 | 1.0884 | 0.9467 |
