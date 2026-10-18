# Iteration 12: results

Retained: ordinary material steps stop the negligible-step scan early, while all-zero/tiny finite steps retain the same threshold. Added a regression proving NaN displacement cannot take the negligible-step shortcut or mutate B.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-12.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.7994 | 13.2492 | 0.8151 |
| kernel | hessian 128 | 2.6832 | 2.7764 | 0.9664 |
| kernel | csr 128 | 0.7653 | 0.8812 | 0.8684 |
| kernel | csr-dense 128 | 4.4963 | 4.4691 | 1.0061 |
| kernel | transpose 128 | 0.5228 | 0.5286 | 0.9890 |
| kernel | kkt 128 | 10.7256 | 13.1661 | 0.8146 |
| kernel | bfgs 128 | 4.9057 | 5.2893 | 0.9275 |
| kernel | cholesky 96 | 3.6779 | 3.6274 | 1.0139 |
| kernel | inverse 96 | 9.1446 | 9.5912 | 0.9534 |
| kernel | qp-dense 64 | 3.3300 | 3.9842 | 0.8358 |
| kernel | qp-pairs 64 | 1.8280 | 1.9344 | 0.9450 |
| kernel | qp-free 128 | 7.3381 | 6.7263 | 1.0910 |
| kernel | qp-box 128 | 7.2860 | 6.8476 | 1.0640 |
| kernel | qp-zero 64 | 1.8016 | 1.8891 | 0.9537 |
| nlp | orbit 20 | 31.7277 | 31.5454 | 1.0058 |
| nlp | orbit 40 | 270.4399 | 276.2387 | 0.9790 |
| nlp | separable 128 | 18.5840 | 17.6090 | 1.0554 |
| nlp | restoration 64 | 0.9801 | 0.8383 | 1.1691 |
