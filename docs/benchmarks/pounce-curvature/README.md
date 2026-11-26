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

The paired tables give both unique F+D points and actual grouped evaluation
attempts. Revisiting a previous point after the one-point adapter cache has
moved incurs new model work. The solver-dual-only ratio restricts the latter
measure to pairs independently certified by both solvers’ own multipliers.
The model timers cover evaluation and validation inside the accounting wrapper.
Returning arrays to the foreign solver and its factorization take additional time.

`--recheck` loads the pinned S2MPJ models and replays starts, multiplier
conversion, residuals and objective comparisons from the saved candidates.
`--check` makes false strict successes, missing or discrepant candidate
objectives, failed release fixtures and infrastructure failures fail CI.
An ordinary solver limit stays visible with its independent verdict.
