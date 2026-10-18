# Iteration 08: results

Retained: singleton Jacobian rows use one direct product, including the original leading-zero addition. Empty, dense and general sparse rows retain their prior paths. Existing restoration and finite-arithmetic tests cover the specialization.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-08.json. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 9.7842 | 10.1167 | 0.9671 |
| kernel | hessian 128 | 3.1307 | 2.9654 | 1.0558 |
| kernel | csr 128 | 0.6856 | 0.9755 | 0.7028 |
| kernel | csr-dense 128 | 4.9592 | 4.7067 | 1.0536 |
| kernel | transpose 128 | 0.5799 | 0.5937 | 0.9768 |
| kernel | kkt 128 | 16.0817 | 16.5475 | 0.9719 |
| kernel | bfgs 128 | 7.6037 | 7.9586 | 0.9554 |
| kernel | cholesky 96 | 3.9995 | 3.6530 | 1.0949 |
| kernel | inverse 96 | 9.3549 | 9.2597 | 1.0103 |
| kernel | qp-dense 64 | 4.0716 | 3.3232 | 1.2252 |
| kernel | qp-pairs 64 | 1.8206 | 2.1000 | 0.8669 |
| kernel | qp-free 128 | 6.8661 | 7.2016 | 0.9534 |
| kernel | qp-box 128 | 7.7432 | 7.0160 | 1.1037 |
| kernel | qp-zero 64 | 1.7308 | 1.7752 | 0.9750 |
| nlp | orbit 20 | 33.7745 | 34.5049 | 0.9788 |
| nlp | orbit 40 | 266.1249 | 266.8070 | 0.9974 |
| nlp | separable 128 | 17.5799 | 17.1817 | 1.0232 |
| nlp | restoration 64 | 0.8784 | 0.8447 | 1.0399 |

## Final release refinement

The final common-harness comparison exposed a two-entry sparse-row regression from the additional dispatch branches. The 0.15 release adds a direct two-term sum as the corresponding short-row path, retaining the original arithmetic order. The slower pre-refinement common measurements are preserved as final-common-before-shortrow.json. Full numerical tests, paired independent validation and final timings were rerun for the refined release; see RESULTS.md.
