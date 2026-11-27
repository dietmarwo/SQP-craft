# Review of iterations 61 to 90

27 November 2026. The delivery is `SQP-craft-iterations-61-90-patch.zip`
(6,984,570 bytes, SHA-256
`5eb414ffd6f1e449e911b1dbdc65843a03a6a7b70f13291ee5ea9e8622562aa2`). Its
production patch `SQP-craft-performance.patch` (SHA-256
`7b34131bcde441324fad6b4fa91610a09c530ed35b381ab4eb6af87ad7a8aae5`)
applied unchanged to `d6b28ca`; that is the first commit of the pull
request. [RESULTS.md](RESULTS.md) is the delivered report, left as
delivered; [PLAN.md](PLAN.md) is the plan it worked from.
[`data/archive.sha256`](data/archive.sha256) is the archive's own
`SHA256SUMS` (146 entries, all matched at import) and
[`data/campaign-summary.json`](data/campaign-summary.json) its summary of
the paired measurements. The second patch of the archive, with the 26
discarded trials, is not imported.

**The four retained options do what the report says, and defaults are
unchanged.** Two of the report's conclusions need a correction, both in
the options' favour: shift bracketing holds up on 240 problems it was not
tuned on, and the two watchdog losses that kept the tiled kernel opt-in
are not caused by the kernel.

## What was measured

Release builds of the base and of the patched tree were made here from
source (two cores, one numerical thread).

| Check | Delivered | Here |
| --- | --- | --- |
| Rust tests, Python tests | 237, 263 | 237, and 257 with 6 skipped for missing optional comparators; 239 and 258 after this review |
| Defaults against the base | not reported | the 20-problem curvature panel with `auto` and `clarabel+exact`: all 40 records identical apart from times and the three new option fields |
| Exact Hessians with bracketing, same panel | 19 to 20 of 20 verified; HS38 500 to 65 steps, HS71 56 to 38, DTOC2 11 to 7, CATENARY 20 to 22 | the same |
| Transfer problem, local initial scaling | 75 to 40 steps (26 arcs), 108 to 57 (51), 127 to 78 (101); 1.59 to 2.00 times faster | the same steps; 1.78, 1.80 and 1.65 times faster; offset start at 26 arcs 81 to 39 |
| Transfer problem, scaling | `jacobian` ends in `qp_failure` at step 6 (51 arcs); `elements` verifies in 159 steps | the same, stationarity 7.76e-7; 26 arcs: `qp_failure` at step 19 against 117 steps |
| Tiled inverse | bit-identical at six sizes | the same test passes; kernel times below |

### Shift bracketing outside the tuning panel

The report's evidence for `exact_shift_bracketing` is the 20-problem panel
on which the trial was developed. The review drew 60 problems from each of
the four subsets of the frozen 1,036-problem selection, excluding that
panel (seed 20261127, list in
[`data/review-bracket-holdout-problems.txt`](data/review-bracket-holdout-problems.txt)),
and ran `clarabel+exact` with and without the option under a 60 s watchdog
(journal: [`data/review-bracket-holdout.jsonl.gz`](data/review-bracket-holdout.jsonl.gz)).

| | Plain | Bracketed |
| --- | ---: | ---: |
| Verified of 240 | 168 | 182 |
| Gained, lost | | 15, 1 |
| Same steps and objective | 143 | |
| Steps where both verified and differed (56 problems) | 3,358 | 1,665 |
| Fewer steps, more steps | | 38, 7 |

Twelve of the gains were at the 500-step limit without the option and
three at the watchdog. The loss is DEVGLA2: 13 steps plain, the step limit
bracketed. The option is not free of risk, but on this sample it is the
better exact mode by a wide margin.

### The tiled kernel and the two watchdog losses

The report kept `tiled_dense_inverse` opt-in because a sweep with tiling
always on verified 812 instead of 814 problems: DRCAV3LQ and DUAL2 reached
the watchdog. Results are bit-identical, so only time can differ. Here:

| Variables | Column kernel | Tiled | Ratio |
| ---: | ---: | ---: | ---: |
| 80 | 50.0 µs | 25.2 µs | 0.50 |
| 96 | 76.5 µs | 42.4 µs | 0.55 |
| 128 | 153.2 µs | 98.8 µs | 0.65 |
| 164 | 283.1 µs | 201.8 µs | 0.71 |
| 196 | 454.3 µs | 346.7 µs | 0.76 |
| 256 | 934.7 µs | 776.1 µs | 0.83 |

| Problem | Variables | Steps | `active-set` | `active-set+tiled` |
| --- | ---: | ---: | ---: | ---: |
| DRCAV3LQ | 196 | 461 | 30.5 s | 30.2 s |
| DUAL2 | 96 | 113 | 37.4 s | 38.1 s |
| DUAL1 | 85 | 78 | 18.5 s | 18.2 s |

The kernel is faster at every size, and the two problems take as long with
it as without: the inverse factor is a small part of these solves. Both
need tens of seconds, so on a loaded host either arm can cross a fixed
watchdog. The losses are a timing effect of problems near the deadline,
not a cost of tiling. The option stays opt-in in this change; nothing
measured here speaks against enabling it by default later.

## Findings and changes

1. **The job schema and the worker disagreed on the new job field.** The
   shared job-API harness, which CI runs and the delivery did not, failed:
   the worker reads an absent or `null` `partitioned_initial_scaling` as
   off, while the published schema allowed a Boolean only. The schema now
   allows `null`, as it does for `elastic_mode`, and states the cross-field
   rule the worker enforces: `true` requires `hessian: "partitioned"`.
2. **No benchmark arm existed for the new options**, so the repository's
   runner could not repeat the report's comparisons (the archive's scripts
   register arms at run time). `clarabel+exact+bracket` and
   `active-set+tiled` are now in `bench/solvers.py`.
3. **The shift search had no unit test.** Two were added: the bracketed
   shift of a 2×2 indefinite matrix lies between the needed and the
   row-sum shift and leaves the smallest eigenvalue above the margin; a
   diagonal matrix, where the row sums are exact, declines the search.
4. **The reference documentation did not list the options.** The option
   tables of `docs/api.md` and `ai-context.md` now have them, and
   `docs/structured-sqp.md` states what each measured, including that
   `scaling="elements"` needs more steps than no scaling.
5. **The error message for an unknown scaling lacked a comma.**
6. **The mode requirements of the options were untested.** One Python test
   covers the six rejected combinations.

## Not verified here

- The full 1,036-problem sweeps, the dense native timing panels, the AC
  power-flow cases (EPIGRIDS) and the 263-arc target.
- The 26 discarded trials and the probes patch.

## Open points

- Enabling `tiled_dense_inverse` by default needs one clean sweep with a
  watchdog that DRCAV3LQ and DUAL2 do not sit on.
- `exact_shift_bracketing` is a candidate default for exact Hessians. The
  full selection, and DEVGLA2 in particular, should be examined first.
- The job worker accepts `partitioned_initial_scaling` only. It has no
  exact-Hessian or scaling fields, so the other options do not apply there.
