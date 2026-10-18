# Iteration 24: results

Retained: member/multiplier storage reserves the known independent-basis bound once and then reuses capacity. No extra mathematical state or shared storage is introduced. Warm-set ordering, dimension changes and iteration budgets retain their tests.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-24.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 13.2857 | 11.4115 | 1.1642 |
| kernel | hessian 128 | 3.2333 | 2.9723 | 1.0878 |
| kernel | csr 128 | 0.7600 | 0.8277 | 0.9181 |
| kernel | csr-dense 128 | 5.4798 | 5.0592 | 1.0831 |
| kernel | transpose 128 | 0.5940 | 0.5672 | 1.0472 |
| kernel | kkt 128 | 13.6248 | 10.7445 | 1.2681 |
| kernel | bfgs 128 | 4.8220 | 5.0928 | 0.9468 |
| kernel | cholesky 96 | 3.7930 | 3.7508 | 1.0113 |
| kernel | inverse 96 | 9.2148 | 9.6670 | 0.9532 |
| kernel | qp-dense 64 | 3.3209 | 3.1543 | 1.0528 |
| kernel | qp-pairs 64 | 0.7214 | 0.7558 | 0.9546 |
| kernel | qp-free 128 | 1.2575 | 1.2060 | 1.0427 |
| kernel | qp-box 128 | 1.3784 | 1.3218 | 1.0428 |
| kernel | qp-zero 64 | 0.6970 | 0.6543 | 1.0653 |
| kernel | qp-free-rows 128 | 2.1967 | 2.0353 | 1.0793 |
| kernel | qp-fixed 64 | 0.6463 | 0.6734 | 0.9597 |
| kernel | qp-dense-normals 64 | 4.5078 | 4.5528 | 0.9901 |
| nlp | orbit 20 | 29.2602 | 28.9502 | 1.0107 |
| nlp | orbit 40 | 246.7924 | 248.9326 | 0.9914 |
| nlp | separable 128 | 16.5192 | 15.9077 | 1.0384 |
| nlp | restoration 64 | 0.7795 | 0.8279 | 0.9415 |
