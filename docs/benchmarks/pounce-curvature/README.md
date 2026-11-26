# POUNCE curvature pilot

The [frozen selection](selection.json) has 20 cases from the standard S2MPJ
benchmark. Each case runs from the clipped S2MPJ start through SQP-craft's
default full BFGS and the active-set SQP engine in POUNCE 0.12.0, with its
exact and L-BFGS modes. Four cases also run POUNCE damped BFGS and the two
SQP-craft Clarabel arms. The latter use full BFGS and the existing exact
Lagrangian Hessian callback with the same QP configuration.

The benchmark imports the separately installed POUNCE wheel. POUNCE is
EPL-2.0 and the adapter uses its public Python interface. The exact arms
use a complete lower triangle for these small cases. At the pinned tag,
POUNCE's SQP L-BFGS implementation expands its history to a dense Hessian
for the QP. This panel measures behavior on small NLPs. [Pinned POUNCE SQP guide](https://github.com/jkitchin/pounce/blob/v0.12.0/docs/src/active-set-sqp.md)

## Results

[tables.md](tables.md) is the generated report of the committed records
([primary](primary.jsonl.gz), [controls](controls.jsonl.gz),
[fixtures](fixtures.json)); [native-exact](native-exact.jsonl.gz) is the
supplemental SQP-craft Clarabel exact arm on all 20 cases. Verified means that
the fresh first-order KKT check passed. Strict means that the solver also
claimed convergence.

| Run | Verified | Strict and verified | Supplied duals (fixed variables completed) | Recovered duals | False strict claims |
|---|---:|---:|---:|---:|---:|
| SQP-craft default BFGS | 20/20 | 20 | 20 (0) | 0 | 0 |
| POUNCE exact | 19/20 | 18 | 18 (4) | 1 | 0 |
| POUNCE L-BFGS | 20/20 | 20 | 19 (5) | 1 | 0 |
| POUNCE damped BFGS control | 4/4 | 4 | 4 (1) | 0 | 0 |
| SQP-craft Clarabel BFGS control | 4/4 | 4 | 4 (0) | 0 | 0 |
| SQP-craft Clarabel exact control | 4/4 | 4 | 4 (0) | 0 | 0 |
| Supplemental SQP-craft Clarabel exact | 19/20 | 19 | 19 (0) | 0 | 0 |

- **POUNCE exact against L-BFGS.** On the 19 pairs that both verify, the
  median ratio of first-order work is 0.426, with a median of seven Hessian
  calls. On the 18 pairs that both solvers certify with their own multipliers
  it is 0.491; on the 14 of those without a fixed variable it is 0.750.
- **Multipliers.** JNLBRNG1, BQPGABIM, DTOC1L, DTOC2 and CATENARY have fixed
  variables. POUNCE reports no multiplier for those and its multipliers pass
  on everything else (see the verifier notes below). HS2 is the one recovered
  certificate: POUNCE returns a zero multiplier for an active lower bound.
  There it also ends at the other local solution (4.9412 against 0.0504).
- **Failures.** POUNCE exact stops at its iteration limit on CATENARY, and its
  BRYBND point passes the check at that limit. The supplemental native exact
  arm fails HS38 at 500 iterations with stationarity 0.00908. Before the
  flat-Hessian shift it also failed CATENARY, which now converges in 20
  iterations.
- **Hosts.** The records were made on an Intel Xeon with AVX-512. A run of
  the delivery on an AMD EPYC 9V74 gave the same candidates, statuses and
  counts in all 92 records. With the AVX-512 kernels of NumPy and OpenBLAS
  switched off, three records take another path to the same verdict
  (CATENARY with SQP-craft BFGS and POUNCE L-BFGS, HS100 with POUNCE L-BFGS),
  and the 0.426 becomes 0.419. Counts are therefore reproducible only between
  hosts with the same kernels.

The panel has at most 58 variables and the exact adapters emit a full lower
triangle, so it says nothing about scaling.
[implementation-review.md](implementation-review.md) is the delivered log of
the pilot.

Replay the committed records without either optimizer (the second command
needs the fetched S2MPJ models):

```sh
d=docs/benchmarks/pounce-curvature
python -m bench.pounce_report $d/primary.jsonl.gz $d/controls.jsonl.gz $d/fixtures.json --check | cmp - $d/tables.md
python -m bench.pounce_report $d/primary.jsonl.gz $d/controls.jsonl.gz $d/fixtures.json --check --recheck > /dev/null
```

## Reproduce

Run the manually dispatched POUNCE curvature pilot workflow or use a Python
3.12 virtual environment on Linux x86-64:

```sh
export MATURIN_PEP517_ARGS=--locked
python -m pip install '.[bench,test]' numpy==2.5.5 scipy==1.18.1
python -m pip download --no-deps --only-binary=:all: --require-hashes \
  -r docs/benchmarks/pounce-curvature/pounce-linux-x86_64.txt -d wheels
python -m pip install --no-deps wheels/pounce_solver-0.12.0-*.whl
wheel_sha=$(sha256sum wheels/pounce_solver-0.12.0-*.whl)
export SQP_CRAFT_POUNCE_WHEEL_SHA256="${wheel_sha%% *}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
python -m bench.s2mpj fetch
python -m pytest -q tests/test_bench.py tests/test_bench_pounce.py
out=$(mktemp -d)
selection=docs/benchmarks/pounce-curvature/selection.json
python -m bench.pounce_fixtures > "$out/fixtures.json"
python -m bench.run "$selection" "$out/primary.jsonl" --evidence --jobs 2 \
  --solvers auto,pounce-sqp-exact,pounce-sqp-lbfgs
python -m bench.run "$selection" "$out/controls.jsonl" --evidence --jobs 2 \
  --problems ROSENBR,HS44,HS71,DTOC2 --solvers pounce-sqp-damped,clarabel,clarabel+exact
python -m bench.pounce_report "$out/primary.jsonl" "$out/controls.jsonl" "$out/fixtures.json" --check --recheck \
  > "$out/tables.md"
```

The worker applies a 60-second shared solve deadline and a 90-second
watchdog that includes independent verification. The configured 500 outer
iterations have solver-specific semantics. SQP-craft also enforces its
existing 5,000 model-call cap. The results include solve settings,
configured start hashes, final points and multiplier vectors, and
fresh original-model KKT and objective checks. The report preserves
the point's independently verified status even if the solver stopped at a
limit or the checker recovered a certificate by least squares.

F/D are distinct points where the grouped value or derivative evaluator
ran. Raw requests and actual grouped evaluations are in `callback_work`.
POUNCE exposes split requests for f, c, gradient and Jacobian. A one-point
cache per group avoids evaluating both halves twice. H is the number of
analytic Hessian call attempts. Separate timers measure the whole Hessian
callback and its S2MPJ model calls. Solver model work excludes setup and
independent verification. Use a fresh file for a different configuration.

The verifier evaluates the full S2MPJ Jacobian at the returned point. When
POUNCE supplies lower and upper bound duals, it tests each side for sign,
presence of a bound and complementarity before crediting a solver certificate.
The report shows the accepted points that needed multiplier recovery.

POUNCE's SQP removes fixed variables and returns zero for both of their bound
multipliers. A fixed variable admits a multiplier of either sign, so for a
zero multiplier of a fixed variable the verifier takes the value that
stationarity requires and then judges the solver's multipliers on every other
variable and row. Such points count under supplied duals, and the report
lists how many there are. A nonzero multiplier of a fixed variable is checked
as supplied, and an active bound of a variable that is not fixed gets no such
completion.

The paired tables give both unique F+D points and actual grouped evaluation
attempts. Revisiting a previous point after the one-point adapter cache has
moved incurs new model work. The solver-dual-only ratio restricts the latter
measure to pairs independently certified by both solvers’ own multipliers.
The model timers cover evaluation and validation inside the accounting wrapper.
Returning arrays to the foreign solver and its factorization take additional time.

`--recheck` loads the pinned S2MPJ models and replays starts, multiplier
conversion, residuals and objective comparisons from the saved candidates.
`--check` makes false strict successes, missing or discrepant candidate
objectives, failed release fixtures, infrastructure failures and a missing
or different POUNCE wheel hash fail CI. The hash is the one exported as
`SQP_CRAFT_POUNCE_WHEEL_SHA256` before the run.
An ordinary solver limit stays visible with its independent verdict.
