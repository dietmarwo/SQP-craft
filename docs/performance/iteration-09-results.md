# Performance iteration 09 results

Version 0.13.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 235.706 | 259.854 | 0.907 | 104 |
| orbit 40 | 1800.575 | 1948.318 | 0.924 | 167 |
| separable 128 | 113.552 | 131.234 | 0.865 | 21 |
| restoration 64 | 5.090 | 5.435 | 0.936 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
