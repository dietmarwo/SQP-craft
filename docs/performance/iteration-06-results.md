# Performance iteration 06 results

Version 0.10.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 249.358 | 267.071 | 0.934 | 104 |
| orbit 40 | 1768.581 | 1897.177 | 0.932 | 167 |
| separable 128 | 115.737 | 120.533 | 0.960 | 21 |
| restoration 64 | 6.149 | 6.261 | 0.982 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
