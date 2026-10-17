# Performance iteration 10 results

Version 0.14.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 259.854 | 247.395 | 1.050 | 104 |
| orbit 40 | 1948.318 | 1832.526 | 1.063 | 167 |
| separable 128 | 131.234 | 115.414 | 1.137 | 21 |
| restoration 64 | 5.435 | 5.914 | 0.919 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.

Final validation subsequently refined small/block Hessian safeguards and added
an explicitly reported block-to-full fallback. See the final results rather
than treating this intermediate checkpoint as the release. Default full-mode
numerical behavior and acceptance thresholds are preserved.
