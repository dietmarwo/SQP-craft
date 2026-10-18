# Iteration 05: results

Retained: each symmetric product row uses one local running sum and a hoisted input scalar. Stored lower-triangle terms and their summation/scatter order are unchanged; full numerical and malformed-input regression tests pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-05.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.0326 | 9.8969 | 1.0137 |
| kernel | hessian 128 | 12.6277 | 3.3225 | 3.8006 |
| kernel | csr 128 | 0.4668 | 0.4683 | 0.9966 |
| kernel | csr-dense 128 | 5.5663 | 5.4483 | 1.0217 |
| kernel | transpose 128 | 0.5280 | 0.5208 | 1.0138 |
| kernel | kkt 128 | 15.5100 | 15.6820 | 0.9890 |
| kernel | bfgs 128 | 7.4679 | 7.0635 | 1.0572 |
| kernel | cholesky 96 | 3.8693 | 3.6211 | 1.0686 |
| kernel | inverse 96 | 9.5703 | 9.5822 | 0.9988 |
| kernel | qp-dense 64 | 3.9040 | 3.2397 | 1.2051 |
| kernel | qp-pairs 64 | 1.7814 | 1.6932 | 1.0521 |
| kernel | qp-free 128 | 6.9136 | 6.7767 | 1.0202 |
| kernel | qp-box 128 | 6.8686 | 6.8032 | 1.0096 |
| kernel | qp-zero 64 | 1.7064 | 2.0073 | 0.8501 |
| nlp | orbit 20 | 33.5546 | 31.3117 | 1.0716 |
| nlp | orbit 40 | 276.7108 | 259.4990 | 1.0663 |
| nlp | separable 128 | 19.5074 | 16.8959 | 1.1546 |
| nlp | restoration 64 | 1.0111 | 0.8947 | 1.1301 |
