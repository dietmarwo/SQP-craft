# Iteration 15: results

Retained: a validated fully stored Hessian writes every lower entry exactly once. Incomplete patterns still clear all missing entries. Model/pattern and SPD checks remain mandatory; repeated varying-size and missing-diagonal tests pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-15.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 12.5617 | 13.2943 | 0.9449 |
| kernel | hessian 128 | 3.3481 | 3.0992 | 1.0803 |
| kernel | csr 128 | 0.9271 | 0.9306 | 0.9962 |
| kernel | csr-dense 128 | 5.0170 | 4.7521 | 1.0557 |
| kernel | transpose 128 | 0.6580 | 0.5246 | 1.2543 |
| kernel | kkt 128 | 13.8644 | 15.9508 | 0.8692 |
| kernel | bfgs 128 | 4.9120 | 4.8645 | 1.0098 |
| kernel | cholesky 96 | 3.6109 | 3.8235 | 0.9444 |
| kernel | inverse 96 | 10.1027 | 9.7431 | 1.0369 |
| kernel | qp-dense 64 | 3.3307 | 3.2010 | 1.0405 |
| kernel | qp-pairs 64 | 2.0081 | 1.7083 | 1.1755 |
| kernel | qp-free 128 | 6.9978 | 6.7800 | 1.0321 |
| kernel | qp-box 128 | 6.6509 | 7.1429 | 0.9311 |
| kernel | qp-zero 64 | 1.7075 | 1.7420 | 0.9802 |
| nlp | orbit 20 | 30.3547 | 29.0933 | 1.0434 |
| nlp | orbit 40 | 282.9712 | 277.5105 | 1.0197 |
| nlp | separable 128 | 19.3586 | 19.3068 | 1.0027 |
| nlp | restoration 64 | 1.5814 | 0.8413 | 1.8797 |
