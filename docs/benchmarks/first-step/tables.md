# First-step acceptance results

Fresh 0.19.0 automatic baseline: **814 / 1,036 accepted**. Accepted means the unchanged independent first-order check accepts the returned point. No variant makes a false convergence claim.

| Variant | Accepted | Net | Unconstrained | Bounds | Linear | Nonlinear | Median paired actual callback ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto | 814 | +0 | +0 | +0 | +0 | +0 | 1.0000 |
| auto+step | 819 | +5 | +7 | -4 | +0 | +2 | 1.0000 |
| auto+safeguard | 824 | +10 | +8 | -1 | +0 | +3 | 1.0000 |
| auto+safeguard10 | 823 | +9 | +7 | -5 | +0 | +7 | 1.0000 |
| active-set+safeguard | 811 | -3 | +8 | -1 | +1 | -11 | 1.0000 |

## Decision

Keep `major_step_limit=None` and `initial_curvature_scaling=False`. The delta-2 proposed safeguard gains 23 and loses 13, meeting +10 net but losing one net bound-constrained problem. The delta-10 and step-only variants also lose in that subset. The explicit active-set variant is compared with the automatic baseline as an additional policy comparison; it is not a replacement automatic default.

## Compatibility and validation

- 1033/1033 uncensored, previously non-error automatic records match 0.19.0 exactly after timings and new disabled diagnostics are removed. HS25NE's former exception becomes a status. The two wall-limited records retain their status with timing-dependent work counts.
- Backend panel: 212 records, 0 non-timing differences.
- Default stress: 3000 records, 0 non-timing differences after version normalization.
- Enabled safeguard stress: 3000 runs, 0 false convergence claims, 0 lost convergences.
- Status panel: 20 exceptions in the planned 66 baseline runs, 0 after the fix. The augmented-update supplement makes 72 records with 21 baseline exceptions and 0 final exceptions.
- SQP1–SQP6 and X2 numerical tutorial summaries match; orbit/Maratos and job fixtures are covered by the Rust/Python suites.

## Callbacks

The full harness counts distinct model points. `callbacks/` replays commonly accepted pairs and records actual `SolveResult.counts` function and derivative invocations, excluding independent post-solve validation. Both the median per-problem new/old actual-call ratio and the ratio of actual-call medians are checked against 1.02. No tolerance was changed. All failed/slower full-benchmark variants and the rejected early-guard experiment are retained.
