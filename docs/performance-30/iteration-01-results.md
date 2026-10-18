# Iteration 01: results

Retained: valid rows now undergo one ordering pass and a constant-time range check. Invalid out-of-range rows retain a precise offending index; no validation or finite-value gate is removed.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-01.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 15.8649 | 12.5923 | 1.2599 |
| kernel | hessian 128 | 12.3774 | 12.4078 | 0.9975 |
| kernel | csr 128 | 0.4681 | 0.4910 | 0.9532 |
| kernel | csr-dense 128 | 4.7850 | 4.8633 | 0.9839 |
| kernel | transpose 128 | 0.5486 | 0.5359 | 1.0238 |
| kernel | kkt 128 | 17.0180 | 15.0501 | 1.1308 |
| kernel | bfgs 128 | 6.9339 | 6.8737 | 1.0088 |
| kernel | cholesky 96 | 3.5785 | 3.5823 | 0.9989 |
| kernel | inverse 96 | 9.1861 | 9.0948 | 1.0100 |
| kernel | qp-dense 64 | 3.6988 | 3.4889 | 1.0602 |
| kernel | qp-pairs 64 | 1.7447 | 2.0849 | 0.8368 |
| kernel | qp-free 128 | 6.5790 | 6.5842 | 0.9992 |
| kernel | qp-box 128 | 6.7534 | 6.6537 | 1.0150 |
| kernel | qp-zero 64 | 1.7159 | 1.7412 | 0.9855 |
| nlp | orbit 20 | 34.8438 | 35.3416 | 0.9859 |
| nlp | orbit 40 | 287.1703 | 293.6219 | 0.9780 |
| nlp | separable 128 | 20.3422 | 21.0529 | 0.9662 |
| nlp | restoration 64 | 1.0350 | 0.9669 | 1.0703 |
