# Iteration 16: results

Retained: free-row norms are not formed because those rows cannot enter any initial/warm/violated working set. All coefficients remain validated. Three coverage-driven QP fixtures were added and the preceding numerical checkpoint was rebuilt with the identical extended kernel harness before timing.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-16.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 11.2076 | 10.4339 | 1.0742 |
| kernel | hessian 128 | 2.7648 | 2.6770 | 1.0328 |
| kernel | csr 128 | 0.7672 | 0.7701 | 0.9962 |
| kernel | csr-dense 128 | 4.6325 | 4.7208 | 0.9813 |
| kernel | transpose 128 | 0.5425 | 0.5452 | 0.9951 |
| kernel | kkt 128 | 10.8507 | 10.9460 | 0.9913 |
| kernel | bfgs 128 | 4.9972 | 4.9784 | 1.0038 |
| kernel | cholesky 96 | 3.6057 | 3.6361 | 0.9916 |
| kernel | inverse 96 | 9.1721 | 9.0563 | 1.0128 |
| kernel | qp-dense 64 | 3.1746 | 3.1436 | 1.0099 |
| kernel | qp-pairs 64 | 1.7195 | 1.7766 | 0.9679 |
| kernel | qp-free 128 | 6.6789 | 6.7866 | 0.9841 |
| kernel | qp-box 128 | 6.6317 | 6.8352 | 0.9702 |
| kernel | qp-zero 64 | 1.6850 | 1.8793 | 0.8966 |
| kernel | qp-free-rows 128 | 7.4982 | 7.2873 | 1.0289 |
| kernel | qp-fixed 64 | 1.7635 | 1.8534 | 0.9515 |
| kernel | qp-dense-normals 64 | 5.2888 | 5.0376 | 1.0499 |
| nlp | orbit 20 | 29.1441 | 31.2233 | 0.9334 |
| nlp | orbit 40 | 251.5074 | 249.4957 | 1.0081 |
| nlp | separable 128 | 16.6547 | 16.4183 | 1.0144 |
| nlp | restoration 64 | 0.8410 | 0.7934 | 1.0601 |
