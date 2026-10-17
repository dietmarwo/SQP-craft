# Performance iteration 03 results

Version 0.7.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 246.831 | 237.745 | 1.038 | 104 |
| orbit 40 | 1849.573 | 1810.270 | 1.022 | 167 |
| separable 128 | 125.492 | 118.107 | 1.063 | 21 |
| restoration 64 | 6.328 | 6.015 | 1.052 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
