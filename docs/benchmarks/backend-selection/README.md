# Backend-selection evidence

Records behind [automatic backend selection](../../backend-selection.md) and
the elastic-mode rule for problems without a row to relax. They were produced
on 16 November 2026 on an Intel Xeon host at 2.10 GHz (2 CPUs, two solves in
parallel) with CPython 3.12.3, NumPy 2.5.5, SciPy 1.18.1 and S2MPJ
`fea6a70048`: the release candidate at SQP-craft `e8a4386` and the 0.18.0
baseline at `42512b0`. Timings are host-dependent; statuses, counts and
residuals are the comparable quantities.

| File | Contents |
| --- | --- |
| `selection.json` | The 53-problem panel, a subset of the [full selection](../selection.json) with the same metadata. |
| `selection-rationale.json` | The targeted categories of the panel, from the stored 0.14.0 results. |
| `results.jsonl.gz` | `bench.run` records at `e8a4386`: Clarabel, active-set, active-set with elastic mode, and `auto`, each with the independent `check`. |
| `baseline-0.18.0.jsonl.gz` | The same panel at 0.18.0 for the three explicit configurations. |
| `tables.md` | `bench.report` tables of `results.jsonl.gz`. |
| `stress.json` | The automatic-default stress campaign: per family and seed, the tallies and false claims. |

## The panel

"Solved" below means converged and accepted in the stored 0.14.0 results of
the full benchmark. The panel holds:

- the 20 Hock–Schittkowski problems of the CI benchmark job;
- all 10 problems Clarabel solved and plain active-set did not;
- all 6 problems plain active-set solved and elastic mode did not;
- the first eight, alphabetically, of the 17 problems elastic mode solved
  and plain active-set did not, and of the 53 problems plain active-set
  solved and Clarabel did not;
- ALLINITA and ALLINITC.

`selection-rationale.json` lists the four targeted categories; ENGVAL2NE is
in two of them. The panel concentrates on backend disagreements and
regressions, so it does not estimate success rates. No problem was dropped
after a run.
The budgets are those of the full benchmark: 500 accepted steps, 5,000
callbacks, 60 s per solve and a 90 s watchdog. No run timed out.

| Configuration | Accepted | Converged and accepted | False claims |
| --- | --- | --- | --- |
| Clarabel, explicit | 37 | 34 | 0 |
| active-set, explicit | 41 | 38 | 0 |
| active-set with elastic mode, explicit | 41 | 40 | 0 |
| `auto` (the Python default) | 41 | 40 | 0 |
| active-set with elastic mode at 0.18.0 | 38 | 37 | 0 |

"Accepted" means the independent first-order check accepts the returned
point; "converged" is the solver's own claim.

- **Compatibility.** With the timing fields removed, every Clarabel and
  plain active-set record equals its 0.18.0 record. Elastic-mode records
  are unchanged on the 47 problems with constraint rows. On the six without
  rows (three unconstrained, three bound-only), elastic mode now runs the
  plain QP and its records equal plain active-set's. Three of them changed
  status: BROWNDEN and HIMMELBF (`line_search_failure` at 0.18.0) and
  PENALTY2 (`iteration_limit`) converge. Every `auto` record equals the
  explicit active-set-with-elastic record except for the configuration name.
- **Trade-off.** Against plain active-set, `auto` gains BIGGS6NE, BROWNBSNE,
  BYRDSPHR, CYCLOOCF, DEVGLA2NE, DISCS, EXTROSNBNE and POWELLSQ, and loses
  ALLINITA, ALLINITC, FLETCHER, MODBEALENE, SPANHYD, SPINOP, TENBARS3 and
  TRO6X2. FLETCHER and MODBEALENE end `locally_infeasible`, a local result
  and no proof of infeasibility. Plain active-set remains one
  `SqpOptions(elastic_mode=False)` away.
- **Full benchmark.** The policy rests on the stored 1,036-problem results
  of 0.14.0 ([tables](../tables.md)): 754 accepted with Clarabel, 799 with
  active-set and 813 with active-set and elastic mode. There the plain and
  elastic paths coincided on 387 of the 389 problems without rows, with
  equal counts, so the new rule does not change those totals.

## Stress campaign

`stress.json` holds 3,000 automatic runs, all with `qp_solver="auto"` and
`elastic_mode=None`:

- the random family `examples/stress_random.py`, seeds 1–4, 300 problems
  each, run twice (the harness's default and restoration-off
  configurations, which follow the same elastic path): 1,187 converged and
  13 `locally_infeasible` per configuration, every convergence claim
  accepted by the independent check;
- the complementarity family `examples/stress_mpcc.py`, seeds 1–2, 300
  problems each: 600 converged and accepted.

There were no false claims. The file also tallies the harness's SciPy SLSQP
comparison on the random family (1,163 of 1,200 accepted on this host); those
counts depend on the host's numerical libraries.

## Reproducing

From the repository root, in a virtual environment with the package built
(`python -m maturin develop --release --locked`):

```sh
python -m bench.s2mpj fetch
python -m bench.run docs/benchmarks/backend-selection/selection.json panel.jsonl \
  --solvers clarabel,active-set,active-set+elastic,auto --jobs 2
python -m bench.report docs/benchmarks/backend-selection/selection.json panel.jsonl tables.md
```

The stress tallies:

```python
import sys
sys.path.insert(0, "examples")
import stress_mpcc, stress_random

for module, seeds in ((stress_random, range(1, 5)), (stress_mpcc, range(1, 3))):
    for seed in seeds:
        tally, false_claims = module.run(seed, 300, qp_solver="auto", elastic=None)
        print(module.__name__, seed, dict(tally), len(false_claims))
```

Read the compressed records with, for example:

```python
import gzip, json
records = [json.loads(line) for line in gzip.open("results.jsonl.gz", "rt")]
```
