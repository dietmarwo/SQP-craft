# Iteration 29: results

Retained: a complete owned bitwise Hessian key permits factor reuse only after a successful SPD check. Value, signed-zero, pattern, dimension, invalid-input, failed-factor and cloned-backend cases are tested. Compatible reuse is reported as an update; setup+update accounting replaces the former one-factorization-per-QP assertion. The common harness now also covers coupled free QPs.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-29.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.1399 | 10.4229 | 0.9729 |
| kernel | hessian 128 | 2.7645 | 2.7901 | 0.9908 |
| kernel | csr 128 | 0.7948 | 0.7836 | 1.0143 |
| kernel | csr-dense 128 | 5.2053 | 5.1239 | 1.0159 |
| kernel | transpose 128 | 0.5364 | 0.5403 | 0.9928 |
| kernel | kkt 128 | 11.0058 | 11.6078 | 0.9481 |
| kernel | bfgs 128 | 5.1086 | 5.2274 | 0.9773 |
| kernel | cholesky 96 | 3.7481 | 3.8745 | 0.9674 |
| kernel | inverse 96 | 9.5682 | 9.2768 | 1.0314 |
| kernel | qp-dense 64 | 3.5512 | 3.3548 | 1.0585 |
| kernel | qp-pairs 64 | 1.0310 | 0.6889 | 1.4965 |
| kernel | qp-free 128 | 1.4230 | 0.7792 | 1.8263 |
| kernel | qp-box 128 | 0.7042 | 0.2409 | 2.9235 |
| kernel | qp-zero 64 | 0.2484 | 0.1340 | 1.8538 |
| kernel | qp-free-rows 128 | 1.4118 | 0.8937 | 1.5797 |
| kernel | qp-fixed 64 | 0.2467 | 0.1314 | 1.8782 |
| kernel | qp-dense-normals 64 | 4.1464 | 4.2785 | 0.9691 |
| kernel | qp-free-dense 128 | 4.8877 | 2.0195 | 2.4203 |
| nlp | orbit 20 | 30.4401 | 31.8569 | 0.9555 |
| nlp | orbit 40 | 275.9100 | 269.6230 | 1.0233 |
| nlp | separable 128 | 16.6780 | 16.9261 | 0.9853 |
| nlp | restoration 64 | 0.6680 | 0.6816 | 0.9801 |
