# Performance iteration 02 results

Version 0.6.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 247.510 | 246.831 | 1.003 | 104 |
| orbit 40 | 1906.461 | 1849.573 | 1.031 | 167 |
| separable 128 | 125.094 | 125.492 | 0.997 | 21 |
| restoration 64 | 6.873 | 6.328 | 1.086 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
