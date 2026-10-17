# Performance iteration 08 results

Version 0.12.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 262.951 | 235.706 | 1.116 | 104 |
| orbit 40 | 1902.577 | 1800.575 | 1.057 | 167 |
| separable 128 | 129.164 | 113.552 | 1.137 | 21 |
| restoration 64 | 5.289 | 5.090 | 1.039 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
