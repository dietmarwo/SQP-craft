# Iteration 22: results

Retained: polishing and full-step calculations omit absolute-magnitude accumulation that they never consume. Violation tests still compute and use the original rounding-error scale, so numerical feasibility thresholds are unchanged.

Full release Rust workspace tests passed before timing. Same benchmark sources, inputs, fixed work and numerical tolerances; all NLP benchmark convergence/objective/KKT assertions passed. Every sample and binary hash is in data/iteration-22.json.gz. Timings below are medians of three samples after one warmup, with serial rotated/reversed processes pinned to one allowed CPU. Small changes may be timing noise; work reductions and final paired comparisons govern claims.

| Kind | Fixture | Previous ms | Candidate ms | Previous/candidate |
|---|---|---:|---:|---:|
| kernel | validate 128 | 9.6974 | 10.0097 | 0.9688 |
| kernel | hessian 128 | 2.8659 | 2.7368 | 1.0472 |
| kernel | csr 128 | 0.7632 | 0.8236 | 0.9267 |
| kernel | csr-dense 128 | 4.7780 | 4.4568 | 1.0721 |
| kernel | transpose 128 | 0.5311 | 0.5115 | 1.0383 |
| kernel | kkt 128 | 10.5065 | 10.4890 | 1.0017 |
| kernel | bfgs 128 | 4.8751 | 4.8259 | 1.0102 |
| kernel | cholesky 96 | 3.6179 | 3.6658 | 0.9869 |
| kernel | inverse 96 | 9.2655 | 9.1751 | 1.0099 |
| kernel | qp-dense 64 | 3.0959 | 3.2345 | 0.9571 |
| kernel | qp-pairs 64 | 0.7534 | 0.6842 | 1.1011 |
| kernel | qp-free 128 | 1.3022 | 1.2172 | 1.0699 |
| kernel | qp-box 128 | 1.5029 | 1.3393 | 1.1222 |
| kernel | qp-zero 64 | 0.6482 | 0.6578 | 0.9854 |
| kernel | qp-free-rows 128 | 2.2127 | 2.0428 | 1.0832 |
| kernel | qp-fixed 64 | 0.7001 | 0.6430 | 1.0889 |
| kernel | qp-dense-normals 64 | 4.2584 | 4.1843 | 1.0177 |
| nlp | orbit 20 | 29.4746 | 28.1417 | 1.0474 |
| nlp | orbit 40 | 251.0595 | 245.3783 | 1.0232 |
| nlp | separable 128 | 16.0978 | 15.7892 | 1.0195 |
| nlp | restoration 64 | 0.7098 | 0.7154 | 0.9921 |
