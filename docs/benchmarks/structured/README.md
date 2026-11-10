# Structured SQP evidence

Raw records behind the measurements in [structured SQP](../../structured-sqp.md)
and the [target card](../../target-card.md). They were produced on 10 November
2026 from SQP-craft `3b053ec` and pykep-core-rust `754701b` with the
structured-SQP and multiple-shooting changes applied (later merged as
SQP-craft 0.18.0 and pykep-rust 0.1.6), on a Debian 13 AMD EPYC host with
CPython 3.12, NumPy 2.3.5, SciPy 1.17.0 and one numerical thread. Timings
are host-dependent; step counts, factor sizes and residuals are the
comparable quantities.

| File | Contents |
| --- | --- |
| `trajectory.jsonl.gz` | One record per `bench/trajectory_scale.py` run (58), with its source file name in `file`: the SQP5 single leg and chain at 7–263 arcs, full and partitioned BFGS, both scalings, both throttle floors, offset starts, the SLSQP and Uno comparators, and the derivative checks. Each solve keeps its point, multipliers, the independent first-order `check` and the DOP853 `replay`. SQP records without a `scaling` field predate that field and used the then-default `scaling="jacobian"` (`bench/trajectory_scale.py` now defaults to `none`). |
| `campaign.jsonl.gz` | The 12 experiment-craft command jobs of `bench/trajectory_campaign.py`: each step's sealed `result.json` with its native receipt (output size and SHA-256), and the campaign status. |
| `s2mpj-structured.jsonl` | The [structured S2MPJ selection](../structured-selection.json) with partitioned SQP and the Uno 2.9.3 filterSQP and IPOPT presets, starting with an environment record. |
| `exact-hessian-tutorials.json` | Output of `tutorials/_support/exact_hessian.py`: SQP4 and SQP6 with full BFGS and the exact Hessian. |

Read a compressed file with, for example:

```python
import gzip, json
records = [json.loads(line) for line in gzip.open("trajectory.jsonl.gz", "rt")]
```

A rerun of the 7-, 13- and 101-arc partitioned cases, the 51-arc offset start,
the 1,916-variable derivative check and the exact-Hessian script on a
different host (Rust 1.97, CPython 3.13, NumPy 2.5.4, SciPy 1.18.1) returned
bit-identical points with the same step counts, factor sizes and replay
defects, and the same exact-Hessian step and shift counts.
