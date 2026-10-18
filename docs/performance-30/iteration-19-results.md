# Iteration 19: results

Retained: exact zero support avoids scanning unaffected Householder columns. The tail norm keeps its original reduction order; nonzero column updates and their order are unchanged. Reflector consistency, orthogonality and dense/random QP checks pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-19.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 12.7774 | 10.3512 | 1.2344 |
| kernel | hessian 128 | 2.8526 | 2.7011 | 1.0561 |
| kernel | csr 128 | 0.8236 | 0.7940 | 1.0373 |
| kernel | csr-dense 128 | 4.4150 | 5.1065 | 0.8646 |
| kernel | transpose 128 | 0.5029 | 0.5832 | 0.8622 |
| kernel | kkt 128 | 12.7209 | 11.0567 | 1.1505 |
| kernel | bfgs 128 | 5.0075 | 4.9386 | 1.0140 |
| kernel | cholesky 96 | 3.9450 | 3.7253 | 1.0590 |
| kernel | inverse 96 | 9.7261 | 9.0733 | 1.0720 |
| kernel | qp-dense 64 | 3.2303 | 3.3231 | 0.9721 |
| kernel | qp-pairs 64 | 1.7796 | 1.8562 | 0.9587 |
| kernel | qp-free 128 | 6.9402 | 7.5515 | 0.9191 |
| kernel | qp-box 128 | 7.4845 | 7.1354 | 1.0489 |
| kernel | qp-zero 64 | 1.7678 | 1.7108 | 1.0333 |
| kernel | qp-free-rows 128 | 7.9085 | 7.3968 | 1.0692 |
| kernel | qp-fixed 64 | 1.6791 | 1.6784 | 1.0004 |
| kernel | qp-dense-normals 64 | 5.0288 | 5.3717 | 0.9362 |
| nlp | orbit 20 | 29.7429 | 30.5269 | 0.9743 |
| nlp | orbit 40 | 258.5838 | 263.6849 | 0.9807 |
| nlp | separable 128 | 16.5198 | 17.5379 | 0.9419 |
| nlp | restoration 64 | 0.8354 | 0.9023 | 0.9258 |
