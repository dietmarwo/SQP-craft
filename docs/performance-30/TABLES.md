# Recomputed performance tables

Baseline is original main 4d776bb, final is 0.15.0. All ratios are baseline/final; larger is faster. No failed solve is counted as a speedup. Single host, repeated fresh processes; raw observations and hashes are adjacent.

## Common pinned baseline/final comparison

| Kind | Fixture | Size | Baseline ms | Final ms | Speedup |
|---|---|---|---|---|---|
| kernel | validate | 128 | 16.9849 | 9.8390 | 1.726 |
| kernel | hessian | 128 | 12.6410 | 2.8536 | 4.430 |
| kernel | csr | 128 | 0.4671 | 0.4869 | 0.959 |
| kernel | csr-dense | 128 | 5.4875 | 4.4301 | 1.239 |
| kernel | transpose | 128 | 0.5075 | 0.5194 | 0.977 |
| kernel | kkt | 128 | 20.7656 | 10.5953 | 1.960 |
| kernel | bfgs | 128 | 7.1939 | 4.7918 | 1.501 |
| kernel | cholesky | 96 | 3.6589 | 3.6929 | 0.991 |
| kernel | inverse | 96 | 9.1353 | 9.3075 | 0.981 |
| kernel | qp-dense | 64 | 3.7037 | 1.7659 | 2.097 |
| kernel | qp-pairs | 64 | 1.7846 | 0.6284 | 2.840 |
| kernel | qp-free | 128 | 6.5720 | 0.8081 | 8.133 |
| kernel | qp-box | 128 | 6.6935 | 0.1956 | 34.223 |
| kernel | qp-zero | 64 | 1.7317 | 0.1301 | 13.314 |
| kernel | qp-free-rows | 128 | 8.3136 | 0.8989 | 9.248 |
| kernel | qp-fixed | 64 | 1.7960 | 0.1340 | 13.399 |
| kernel | qp-dense-normals | 64 | 5.1851 | 4.0034 | 1.295 |
| kernel | qp-free-dense | 128 | 11.5783 | 1.9640 | 5.895 |
| nlp | orbit | 20 | 34.4735 | 28.8821 | 1.194 |
| nlp | orbit | 40 | 282.9428 | 253.0777 | 1.118 |
| nlp | separable | 128 | 19.4280 | 16.1387 | 1.204 |
| nlp | restoration | 64 | 0.9819 | 0.6529 | 1.504 |

## Frozen held-out sizes

| Fixture | Size | Backend | Baseline ms | Final ms | Speedup |
|---|---|---|---|---|---|
| orbit | 30 | active-set | 126.104 | 107.644 | 1.171 |
| orbit | 30 | clarabel | 796.778 | 757.656 | 1.052 |
| orbit | 48 | active-set | 496.656 | 431.830 | 1.150 |
| orbit | 48 | clarabel | 3063.642 | 2940.215 | 1.042 |
| separable | 96 | active-set | 9.436 | 7.891 | 1.196 |
| separable | 96 | clarabel | 59.377 | 57.738 | 1.028 |
| separable | 192 | active-set | 56.613 | 47.714 | 1.187 |
| separable | 192 | clarabel | 297.271 | 302.154 | 0.984 |
| restoration | 48 | active-set | 0.561 | 0.387 | 1.452 |
| restoration | 48 | clarabel | 2.931 | 3.309 | 0.886 |
| restoration | 96 | active-set | 2.241 | 1.714 | 1.308 |
| restoration | 96 | clarabel | 13.324 | 13.208 | 1.009 |

## Cold/reuse and both backends

| Fixture | Size | Backend | QP mode | Baseline ms | Final ms | Speedup | Base BFGS ms | Final BFGS ms |
|---|---|---|---|---|---|---|---|---|
| orbit | 20 | active | reuse | 37.984 | 31.075 | 1.222 | 2.353 | 0.854 |
| orbit | 20 | active | cold | 37.743 | 29.756 | 1.268 | 2.436 | 0.909 |
| orbit | 20 | clarabel | reuse | 230.128 | 244.244 | 0.942 | 2.351 | 0.965 |
| orbit | 20 | clarabel | cold | 265.875 | 260.140 | 1.022 | 2.501 | 0.947 |
| orbit | 40 | active | reuse | 289.768 | 273.524 | 1.059 | 14.312 | 5.631 |
| orbit | 40 | active | cold | 302.839 | 254.371 | 1.191 | 14.655 | 5.475 |
| orbit | 40 | clarabel | reuse | 1762.724 | 1809.341 | 0.974 | 16.247 | 7.657 |
| orbit | 40 | clarabel | cold | 1996.236 | 2008.855 | 0.994 | 16.206 | 7.852 |
| separable | 128 | active | reuse | 20.460 | 16.556 | 1.236 | 1.227 | 0.535 |
| separable | 128 | active | cold | 20.417 | 16.704 | 1.222 | 1.115 | 0.532 |
| separable | 128 | clarabel | reuse | 118.819 | 115.298 | 1.031 | 1.196 | 0.586 |
| separable | 128 | clarabel | cold | 130.604 | 118.533 | 1.102 | 1.211 | 0.622 |
| restoration | 64 | active | reuse | 0.971 | 0.691 | 1.406 | 0.038 | 0.016 |
| restoration | 64 | active | cold | 0.984 | 0.715 | 1.376 | 0.039 | 0.017 |
| restoration | 64 | clarabel | reuse | 5.248 | 5.134 | 1.022 | 0.040 | 0.017 |
| restoration | 64 | clarabel | cold | 5.955 | 5.941 | 1.002 | 0.040 | 0.017 |

## All checkpoint measurements

Each point was remeasured in one shared, rotating, pinned three-repeat campaign. Speedup is the geometric mean of four fixture ratios to checkpoint 00. It is not a product of noisy adjacent speedups. Checkpoints use version 0.14; the final 0.15 release is measured separately above.

| Iteration | Plan | SQP geometric speedup | Orbit-40 BFGS ms |
|---|---|---|---|
| 0 | Frozen baseline | 1.000 | 14.481 |
| 1 | Use the sorted CSR row endpoint for range checks | 1.017 | 14.605 |
| 2 | Check the Hessian triangle through sorted row endpoints | 1.014 | 13.789 |
| 3 | Reuse already validated bound structure in KKT checks | 1.033 | 14.566 |
| 4 | Share checked KKT arithmetic after QP validation | 1.072 | 14.912 |
| 5 | Accumulate symmetric Hessian rows locally | 1.159 | 14.124 |
| 6 | Use contiguous arithmetic for dense lower Hessian rows | 1.135 | 13.708 |
| 7 | Use contiguous products for full CSR rows | 1.117 | 14.420 |
| 8 | Specialize singleton CSR row products | 1.123 | 14.776 |
| 9 | Skip zero multiplier rows after finite-data validation | 1.129 | 14.646 |
| 10 | Fuse finite BFGS validation with triangle construction | 1.134 | 11.844 |
| 11 | Separate contiguous BFGS triangle computation from mirroring | 1.200 | 5.651 |
| 12 | Stop negligible-step scans once a material component is found | 1.141 | 6.020 |
| 13 | Copy exact-curvature differences without damping arithmetic | 1.165 | 5.533 |
| 14 | Clear only the factor triangle that is read | 1.173 | 5.671 |
| 15 | Avoid clearing completely stored Hessian triangles | 1.175 | 5.817 |
| 16 | Avoid unused row norms and extend working-set coverage | 1.182 | 5.338 |
| 17 | Project singleton constraint normals directly | 1.171 | 5.715 |
| 18 | Skip dependent insertions after the basis is full | 1.164 | 5.905 |
| 19 | Restrict Householder matrix updates to exact nonzero support | 1.174 | 5.928 |
| 20 | Build exact diagonal inverse factors directly | 1.230 | 5.482 |
| 21 | Skip zero KKT primal direction coefficients | 1.215 | 5.441 |
| 22 | Compute only the constraint activity when its error scale is unused | 1.226 | 5.368 |
| 23 | Borrow the original gradient during nonelastic polishing | 1.237 | 5.758 |
| 24 | Reserve the retained working set once per dimension | 1.229 | 5.283 |
| 25 | Scan only bounded constraint sources for violations | 1.219 | 5.115 |
| 26 | Reuse a fresh equality-source list across attempts | 1.207 | 5.293 |
| 27 | Solve fully unconstrained QPs from the checked Cholesky factor | 1.218 | 5.243 |
| 28 | Check diagonal box minimizers before general working-set search | 1.266 | 5.317 |
| 29 | Reuse Cholesky factors only for bit-identical Hessians | 1.216 | 5.387 |
| 30 | Cache the unrotated inverse only after confirmed Hessian reuse | 1.261 | 5.117 |
