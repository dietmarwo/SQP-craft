# Iteration 30: results

Retained: a confirmed exact-H repeat promotes unrotated J and its norm to a per-backend cache. Nonrepeating NLP Hessians incur no extra inverse copy. Every H miss/failed factor invalidates the cache; changed gradients/Jacobians/row bounds safely reuse only the H-dependent inverse. Added promotion, changed-row, H-change and failed-factor tests; all original-data acceptance checks remain mandatory.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-30.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 9.5245 | 9.9523 | 0.9570 |
| kernel | hessian 128 | 3.0091 | 2.8629 | 1.0510 |
| kernel | csr 128 | 0.7761 | 0.8185 | 0.9483 |
| kernel | csr-dense 128 | 4.4933 | 4.4491 | 1.0099 |
| kernel | transpose 128 | 0.5532 | 0.5160 | 1.0720 |
| kernel | kkt 128 | 10.5906 | 10.8474 | 0.9763 |
| kernel | bfgs 128 | 4.7899 | 4.9550 | 0.9667 |
| kernel | cholesky 96 | 3.6589 | 3.6590 | 1.0000 |
| kernel | inverse 96 | 9.4719 | 9.4754 | 0.9996 |
| kernel | qp-dense 64 | 2.7810 | 1.9955 | 1.3936 |
| kernel | qp-pairs 64 | 0.7980 | 0.6817 | 1.1706 |
| kernel | qp-free 128 | 0.8769 | 0.8447 | 1.0381 |
| kernel | qp-box 128 | 0.1992 | 0.1922 | 1.0368 |
| kernel | qp-zero 64 | 0.1273 | 0.1264 | 1.0073 |
| kernel | qp-free-rows 128 | 0.9064 | 0.9035 | 1.0032 |
| kernel | qp-fixed 64 | 0.1328 | 0.1315 | 1.0096 |
| kernel | qp-dense-normals 64 | 4.0457 | 4.4855 | 0.9020 |
| kernel | qp-free-dense 128 | 2.0404 | 2.0785 | 0.9817 |
| nlp | orbit 20 | 29.7604 | 30.0344 | 0.9909 |
| nlp | orbit 40 | 262.7186 | 259.3123 | 1.0131 |
| nlp | separable 128 | 16.2433 | 16.4870 | 0.9852 |
| nlp | restoration 64 | 0.6514 | 0.6523 | 0.9987 |
