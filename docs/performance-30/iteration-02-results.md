# Iteration 02: results

Retained: after full CSR validation, the valid lower-triangle gate reads one endpoint per row instead of every stored Hessian index. Invalid models still receive an above-diagonal diagnostic before numerical use.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-02.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 12.2694 | 12.0454 | 1.0186 |
| kernel | hessian 128 | 12.5130 | 12.4564 | 1.0045 |
| kernel | csr 128 | 0.4875 | 0.4620 | 1.0551 |
| kernel | csr-dense 128 | 4.7423 | 4.7853 | 0.9910 |
| kernel | transpose 128 | 0.5637 | 0.5376 | 1.0485 |
| kernel | kkt 128 | 14.8854 | 16.9785 | 0.8767 |
| kernel | bfgs 128 | 6.9834 | 7.1272 | 0.9798 |
| kernel | cholesky 96 | 3.8121 | 3.6421 | 1.0467 |
| kernel | inverse 96 | 9.1968 | 9.4892 | 0.9692 |
| kernel | qp-dense 64 | 3.5738 | 3.5489 | 1.0070 |
| kernel | qp-pairs 64 | 1.7460 | 1.7168 | 1.0170 |
| kernel | qp-free 128 | 7.0428 | 6.7428 | 1.0445 |
| kernel | qp-box 128 | 7.0845 | 6.9297 | 1.0223 |
| kernel | qp-zero 64 | 1.7700 | 1.7461 | 1.0137 |
| nlp | orbit 20 | 33.6027 | 33.4282 | 1.0052 |
| nlp | orbit 40 | 277.7155 | 279.4030 | 0.9940 |
| nlp | separable 128 | 19.5218 | 19.1482 | 1.0195 |
| nlp | restoration 64 | 0.9621 | 0.9350 | 1.0289 |
