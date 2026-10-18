# Iteration 28: results

Retained: exactly diagonal SPD box proposals require full original-row/dual KKT acceptance; failures fall back and elastic QPs are excluded. Added constrained fallback/infeasibility regressions. Initial tests caught obsolete exact work-count expectations: closed-form QPs make zero working-set changes, and the SQP warm-start test now verifies the returned active bounds are reused (cold 6 changes versus warm 0), preserving identical points, outer steps and tolerances. The first failed log is retained.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-28.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.0636 | 11.9116 | 0.8449 |
| kernel | hessian 128 | 2.7067 | 2.7495 | 0.9844 |
| kernel | csr 128 | 0.7900 | 0.8362 | 0.9447 |
| kernel | csr-dense 128 | 4.7065 | 4.7497 | 0.9909 |
| kernel | transpose 128 | 0.5229 | 0.5286 | 0.9893 |
| kernel | kkt 128 | 10.8452 | 12.6824 | 0.8551 |
| kernel | bfgs 128 | 4.8531 | 5.0391 | 0.9631 |
| kernel | cholesky 96 | 3.8942 | 3.7347 | 1.0427 |
| kernel | inverse 96 | 9.4849 | 9.0619 | 1.0467 |
| kernel | qp-dense 64 | 3.1515 | 3.5734 | 0.8819 |
| kernel | qp-pairs 64 | 0.7884 | 0.7968 | 0.9895 |
| kernel | qp-free 128 | 1.2146 | 1.2189 | 0.9965 |
| kernel | qp-box 128 | 1.2915 | 0.6517 | 1.9818 |
| kernel | qp-zero 64 | 0.6417 | 0.2554 | 2.5122 |
| kernel | qp-free-rows 128 | 2.0611 | 1.4827 | 1.3901 |
| kernel | qp-fixed 64 | 0.6531 | 0.2974 | 2.1960 |
| kernel | qp-dense-normals 64 | 4.2728 | 4.2836 | 0.9975 |
| nlp | orbit 20 | 30.5644 | 28.6861 | 1.0655 |
| nlp | orbit 40 | 265.0324 | 251.5466 | 1.0536 |
| nlp | separable 128 | 16.5501 | 16.7508 | 0.9880 |
| nlp | restoration 64 | 0.7641 | 0.6480 | 1.1791 |
