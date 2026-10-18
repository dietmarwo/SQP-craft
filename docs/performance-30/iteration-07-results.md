# Iteration 07: results

Retained: dense CSR rows avoid all column-index gathers, while sparse rows retain the original product. Public length checks, pattern validation at model boundaries and scalar sum order remain in force.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-07.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 12.4104 | 9.9604 | 1.2460 |
| kernel | hessian 128 | 2.8522 | 2.9094 | 0.9804 |
| kernel | csr 128 | 0.4715 | 0.6341 | 0.7436 |
| kernel | csr-dense 128 | 4.7927 | 4.7839 | 1.0018 |
| kernel | transpose 128 | 0.5358 | 0.6191 | 0.8655 |
| kernel | kkt 128 | 17.3738 | 15.2339 | 1.1405 |
| kernel | bfgs 128 | 7.3593 | 7.0495 | 1.0439 |
| kernel | cholesky 96 | 3.6606 | 3.6097 | 1.0141 |
| kernel | inverse 96 | 9.3719 | 10.2543 | 0.9140 |
| kernel | qp-dense 64 | 3.2804 | 3.3981 | 0.9654 |
| kernel | qp-pairs 64 | 1.6908 | 1.9003 | 0.8898 |
| kernel | qp-free 128 | 6.9449 | 6.8677 | 1.0113 |
| kernel | qp-box 128 | 6.9174 | 7.1233 | 0.9711 |
| kernel | qp-zero 64 | 1.7812 | 1.7826 | 0.9992 |
| nlp | orbit 20 | 30.8481 | 32.7350 | 0.9424 |
| nlp | orbit 40 | 272.9574 | 280.1818 | 0.9742 |
| nlp | separable 128 | 17.6259 | 17.8717 | 0.9862 |
| nlp | restoration 64 | 0.8583 | 1.0911 | 0.7866 |
