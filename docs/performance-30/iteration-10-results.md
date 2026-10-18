# Iteration 10: results

Retained: one finite flag per computed triangle entry replaces a separate full-matrix scan. Exact scalar BFGS formulas, atomic swap-on-success behavior and block SPD checks remain unchanged; exact-expression and padded-indefiniteness regressions pass.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-10.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 9.7518 | 12.2743 | 0.7945 |
| kernel | hessian 128 | 2.7380 | 2.8777 | 0.9515 |
| kernel | csr 128 | 0.7663 | 0.8324 | 0.9205 |
| kernel | csr-dense 128 | 4.6817 | 4.6157 | 1.0143 |
| kernel | transpose 128 | 0.5486 | 0.5417 | 1.0128 |
| kernel | kkt 128 | 11.0302 | 12.7364 | 0.8660 |
| kernel | bfgs 128 | 6.9498 | 4.2564 | 1.6328 |
| kernel | cholesky 96 | 3.6597 | 3.5638 | 1.0269 |
| kernel | inverse 96 | 9.6752 | 9.2115 | 1.0503 |
| kernel | qp-dense 64 | 3.4321 | 3.2099 | 1.0692 |
| kernel | qp-pairs 64 | 1.7237 | 1.7730 | 0.9722 |
| kernel | qp-free 128 | 6.5170 | 6.8628 | 0.9496 |
| kernel | qp-box 128 | 6.6158 | 6.7611 | 0.9785 |
| kernel | qp-zero 64 | 1.7547 | 1.7211 | 1.0196 |
| nlp | orbit 20 | 30.8440 | 30.9136 | 0.9977 |
| nlp | orbit 40 | 265.4708 | 261.4670 | 1.0153 |
| nlp | separable 128 | 16.6491 | 17.1118 | 0.9730 |
| nlp | restoration 64 | 0.8430 | 0.8315 | 1.0138 |
