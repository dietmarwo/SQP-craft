# SQP numerical failures and first-step experiments

This evidence accompanies the 0.20.0 source preparation. All convergence claims use the unchanged independent benchmark check. The full selection, budgets and pinned S2MPJ source remain those of `bench/`.

The campaign reruns 0.19.0 as its baseline and compares automatic selection with the step limit alone, the limit with fresh-curvature scaling at delta 2 and delta 10, and explicit active-set with both. The experiment names are in `bench/solvers.py`. Both new controls can be disabled to retain the legacy step policy.

The generated `summary.json` and `tables.md` record the acceptance decision, failures, callback counts, subset results and compatibility differences. Raw records and per-case stress outputs are retained here, including negative experiments. `implementation-review.md` gives the item-by-item review and release order.

## Reproduce

Build the matching Python package with `python -m maturin develop --release --locked` in an isolated environment. Then:

```sh
python -m bench.s2mpj fetch
OPENBLAS_NUM_THREADS=1 python -m bench.run docs/benchmarks/selection.json results.jsonl \
  --solvers auto,auto+step,auto+safeguard,auto+safeguard10,active-set+safeguard --jobs 6
python scripts/stress_weaknesses.py stress.jsonl.gz
python scripts/stress_weaknesses.py stress-safeguard.jsonl.gz --variant safeguard
```

Use a separate baseline checkout and virtual environment at `55195896ab3a0b94afe67bad4c7148dfbe8955bb` for the old `auto` run. The benchmark supervises every solve with a 90 s watchdog and a cooperative 60 s solver budget; failures and censored runs stay in the records.

HS25NE's extracted first QP is in `crates/activeset/tests/fixtures/hs25ne.rs`. It derives from S2MPJ `fea6a70048eaad28b13a08703ddbfdbf65cd9c30` (BSD-3-Clause), with the standard start and no rounded coefficients. The original Householder coefficient squared a norm of order 1e-290 and underflowed, producing an infinite coefficient and NaNs. Extreme tails are now normalized before that product; ordinary-path arithmetic is unchanged.

## Default decision

Both experimental controls remain **off** (`major_step_limit=None`, `initial_curvature_scaling=False`). The proposed delta-2 variant reaches the required +10 net accepted problems but loses one net bound-constrained problem. All tested limit variants lose in that subset. This is a rejected default change, retained as an explicit option for measured use on an application's own problems.

The 5,180 variant records and 1,036 fresh baseline records remain available. The 53-problem backend panel and 3,000 default stress records are paired separately. There are two wall-censored baseline/default solves; they retain `time_limit` but their work counts are naturally timing-dependent. All 1,033 uncensored baseline runs that already returned a status are unchanged. HS25NE's former exception becomes a status. Tutorial numerical summaries agree across SQP1–SQP6 and X2 after removing timings, backend-version labels and new disabled diagnostics.

An implementation review rejected additional intermediate QP guards: an infinite full step can still admit a finite partial or elastic step. `early-guard-experiment.jsonl.gz` preserves that experiment. The final implementation retains the normalized reflector and final backend-output validation, and `guard-correction.jsonl.gz` completely replays the four affected problems across all five configurations. `results.jsonl.gz` incorporates that correction. The source hashes are in `provenance.json`.

The full harness's `functions` and `derivatives` count distinct model points. A secondary commonly-accepted panel records the actual `SolveResult.counts` callback invocations; it is reproduced with:

```sh
python -m bench.callback_panel baseline-0.19.0.jsonl.gz results.jsonl.gz callback-panel \
  --baseline-python /absolute/baseline/.venv/bin/python
python -m bench.first_step_report
```

The baseline interpreter runs the current harness against its own installed 0.19.0 package. Optional diagnostics absent from that package are recorded as null. The secondary records and selection are stored under `callbacks/`.
