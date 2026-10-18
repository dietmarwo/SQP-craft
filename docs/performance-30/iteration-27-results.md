# Iteration 27: results

Retained: nonelastic fully unconstrained QPs use the SPD factor, one original-H refinement and the unchanged independent acceptance gate, with budget checks before/after the proposal. Failed arithmetic/checks fall back to the general solver. Added scaled analytic coupled-SPD tests; removed the now-unused source-count helper.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-27.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.0731 | 10.8574 | 0.9278 |
| kernel | hessian 128 | 2.8413 | 2.8730 | 0.9890 |
| kernel | csr 128 | 0.7422 | 0.7881 | 0.9417 |
| kernel | csr-dense 128 | 5.0158 | 4.6133 | 1.0872 |
| kernel | transpose 128 | 0.5407 | 0.5378 | 1.0054 |
| kernel | kkt 128 | 10.7145 | 10.5480 | 1.0158 |
| kernel | bfgs 128 | 5.1370 | 4.9464 | 1.0386 |
| kernel | cholesky 96 | 3.7772 | 3.6843 | 1.0252 |
| kernel | inverse 96 | 9.2514 | 9.3460 | 0.9899 |
| kernel | qp-dense 64 | 3.1906 | 3.2442 | 0.9835 |
| kernel | qp-pairs 64 | 0.6944 | 0.7686 | 0.9035 |
| kernel | qp-free 128 | 1.2683 | 1.2747 | 0.9950 |
| kernel | qp-box 128 | 1.2872 | 1.3235 | 0.9725 |
| kernel | qp-zero 64 | 0.7230 | 0.6749 | 1.0712 |
| kernel | qp-free-rows 128 | 2.3083 | 2.2466 | 1.0275 |
| kernel | qp-fixed 64 | 0.6422 | 0.7026 | 0.9140 |
| kernel | qp-dense-normals 64 | 4.4748 | 4.5465 | 0.9842 |
| nlp | orbit 20 | 29.8481 | 29.3313 | 1.0176 |
| nlp | orbit 40 | 261.3687 | 259.9512 | 1.0055 |
| nlp | separable 128 | 16.6478 | 16.0905 | 1.0346 |
| nlp | restoration 64 | 0.8235 | 0.8252 | 0.9979 |
