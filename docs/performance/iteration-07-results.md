# Performance iteration 07 results

Version 0.11.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 267.071 | 262.951 | 1.016 | 104 |
| orbit 40 | 1897.177 | 1902.577 | 0.997 | 167 |
| separable 128 | 120.533 | 129.164 | 0.933 | 21 |
| restoration 64 | 6.261 | 5.289 | 1.184 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
