# Performance iteration 04 results

Version 0.8.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 237.745 | 236.098 | 1.007 | 104 |
| orbit 40 | 1810.270 | 1819.159 | 0.995 | 167 |
| separable 128 | 118.107 | 123.622 | 0.955 | 21 |
| restoration 64 | 6.015 | 6.888 | 0.873 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
