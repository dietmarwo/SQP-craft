# Performance iteration 05 results

Version 0.9.0. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.

| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |
| --- | ---: | ---: | ---: | ---: |
| orbit 20 | 236.098 | 249.358 | 0.947 | 104 |
| orbit 40 | 1819.159 | 1768.581 | 1.029 | 167 |
| separable 128 | 123.622 | 115.737 | 1.068 | 21 |
| restoration 64 | 6.888 | 6.149 | 1.120 | 4 |

Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.
