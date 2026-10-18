# Iteration 09: results

Retained: validated finite Jacobian rows with zero multipliers perform no gathers or scatter additions in KKT checking. Raw public CSR transpose semantics remain unchanged; all nonfinite-input and overflow tests still pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-09.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 10.1829 | 10.4163 | 0.9776 |
| kernel | hessian 128 | 2.7541 | 2.7930 | 0.9861 |
| kernel | csr 128 | 0.8268 | 0.7892 | 1.0477 |
| kernel | csr-dense 128 | 4.4745 | 4.6706 | 0.9580 |
| kernel | transpose 128 | 0.5398 | 0.5017 | 1.0760 |
| kernel | kkt 128 | 15.3557 | 10.7249 | 1.4318 |
| kernel | bfgs 128 | 6.9883 | 7.4889 | 0.9332 |
| kernel | cholesky 96 | 3.6761 | 3.6102 | 1.0182 |
| kernel | inverse 96 | 9.1952 | 9.2305 | 0.9962 |
| kernel | qp-dense 64 | 3.3956 | 3.6901 | 0.9202 |
| kernel | qp-pairs 64 | 1.6820 | 1.7049 | 0.9866 |
| kernel | qp-free 128 | 6.7225 | 6.6667 | 1.0084 |
| kernel | qp-box 128 | 7.0132 | 6.6112 | 1.0608 |
| kernel | qp-zero 64 | 1.8123 | 1.6855 | 1.0752 |
| nlp | orbit 20 | 31.4674 | 31.6579 | 0.9940 |
| nlp | orbit 40 | 269.9341 | 261.2456 | 1.0333 |
| nlp | separable 128 | 16.9599 | 16.9275 | 1.0019 |
| nlp | restoration 64 | 0.8389 | 0.8539 | 0.9824 |
