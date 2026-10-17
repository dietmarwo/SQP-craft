# Performance iteration 01 results

Version 0.5.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 257.321 | 247.510 | 1.040 | 104 |
| orbit 40 | 1869.524 | 1906.461 | 0.981 | 167 |
| separable 128 | 130.426 | 125.094 | 1.043 | 21 |
| restoration 64 | 6.323 | 6.873 | 0.920 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
