# EPIGRIDS: checked DC dispatch and an AC regression study

This study uses the 250-bus New England MATPOWER data from
[EPIGRIDS v1.0.0](https://github.com/WISPO-POP/SyntheticElectricNetworkModels/tree/v1.0.0),
CC BY 4.0, by Jonathan Snodgrass and collaborators. The case files are read as
they are and never modified. The three easy cases are operating points on one
topology.

## Model and checks

The constrained literal parser never evaluates MATLAB. An independent
matpowercaseframes loader reads the original file, and the two must agree
exactly.

The sparse AC model has 583 variables with 42 active generators; an offline
generator changes this dimension. Both ends of each branch use apparent-power
limits. A zero/zero angle-limit pair follows MATPOWER's unconstrained
convention. The objective is the dispatch cost divided by 10,000.

The DC LP uses 16 secant intervals per quadratic cost and a separate tangent
lower model. LP-IP-craft and SciPy HiGHS solve identical rows. The original
quadratic costs, the approximation bound and the model difference from AC are
reported separately.

The AC reference is PYPOWER's interior-point OPF. A separate checker rebuilds
the admittance matrices with PYPOWER and replays each final native point
independently of the model callback. A native AC run is **accepted** only if
all of these hold:
- native status `converged`;
- balance and normalized limit violation at most 1e-7 per unit;
- stationarity and complementarity at most 1e-6 in original currency (the
  cost scaling is undone first), with valid dual signs;
- the reference run succeeded and passes the same physical check;
- the cost is at most 0.1% above the reference's.

Every run is capped at 60 seconds, 300 accepted steps and 5,000 callback
calls, with one thread. A pass is a first-order KKT point, not a global or
second-order certificate.

## Results of 17 November 2026

The configuration: partitioned BFGS curvature, an initial Hessian scale of
0.01, the augmented-Lagrangian merit and an internal QP tolerance of 1e-12,
with workspace reuse. It was selected on Winter12Generous only and fixed
before the other cases ran. These are settings of this study, not new
solver defaults. The solver's own acceptance tolerances are unchanged.

**5 of 14 runs accepted**, against 0 of 14 with the solver before the
Clarabel recovery of the same date:

| Case | Accepted | Accepted starts | Not accepted |
| --- | ---: | --- | --- |
| Winter12Generous | 1/3 | reference | source: line-search failure; DC replay: stalled |
| Fall116Generous | 1/3 | reference | source: stalled; DC replay: line-search failure |
| Winter68Generous | 3/3 | source, reference, DC replay | — |
| BaseCase | 0/2 | — | both: restoration limit; no feasible DC LP, so no DC replay |
| Spring13Tight | 0/3 | — | source, DC replay: restoration limit; reference: see below |

Starts:
- **source** is the operating point in the case file;
- **reference** starts at PYPOWER's solution. It is a diagnostic, not an
  independent cold solve;
- **DC replay** starts at a power flow of the DC dispatch.

The runs that were not accepted from the source and DC replay starts mostly
reach balances near 1e-13 and the reference cost, but their stationarity of
4.7e-4 to 12 currency fails the 1e-6 gate.

The reference run for BaseCase and Spring13Tight failed, so no run on those
cases can be accepted. From the reference start, Spring13Tight converged to a
point that passes the physical and first-order checks with a cost 2.9% below
the failed reference's. It stays outside the count.

**Ablation.** The solver before the recovery, with the same settings,
stopped with `qp_failure` after 1 to 11 steps on all three Winter12Generous
starts. Clarabel returned `AlmostSolved` for step QPs that the independent
check then rejected. Without the recovery, the independent check also
rejects the two captured step QPs of the regression tests in
`crates/clarabel`.

The records are in [results/2026-11-17/](results/2026-11-17/):
- `protocol.json`: the run's protocol, including the SHA-256 of the
  executed `run.py`;
- `results.jsonl.gz`: per case, the import identity, derivative check, DC
  and reference results, and for every AC run its status, final point,
  multipliers, checks and report, without per-iteration history.

Run [verify.py](verify.py) to recheck them without running an optimizer. It
recomputes every check from the original case files and compares the
acceptance decisions and residuals with the record.

A second machine, an Intel Xeon at 2.1 GHz, reproduced all 14 AC runs
bit for bit: statuses, steps, final points and acceptance. Only the timings
differed.

Several runs take 40 to 55 of their 60 seconds. On a slower machine, the
time limit can end them earlier and change the result.

## Reproduce

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 pypower==5.1.21 matpowercaseframes==2.1.1 lp-ip-craft==0.30.0
# Build/install this repository (maturin develop --release --locked) first.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/epigrids/test_power.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/epigrids/run.py \
  --data /absolute/SyntheticElectricNetworkModels --output /absolute/new-ac-run \
  --merit augmented-lagrangian --qp-tolerance 1e-12 --initial-hessian-scale 0.01
OPENBLAS_NUM_THREADS=1 python bench/epigrids/verify.py \
  --data /absolute/SyntheticElectricNetworkModels \
  --results bench/epigrids/results/2026-11-17/results.jsonl.gz
```

Without options, `run.py` keeps the solver's defaults: the L1 merit, a QP
tolerance of 1e-10 and an initial Hessian scale of 1. Each output records the
solver options it used, and fresh final KKT residuals separately.

## Earlier result and other arms

- **16 November.** With the solver defaults, the DC LP matched HiGHS on all
  eight feasible LPs across four operating points. Both solvers found the
  BaseCase DC LP infeasible: minimum generation exceeds its load. No native
  AC run was accepted. Clarabel rejected step QPs, usually with
  `AlmostSolved`.
- **Exact Hessian.** The optional `--hessian exact` arm uses sparse analytic
  PYPOWER second derivatives, checked against directional gradient
  differences. Its first reference-start probe ended in a QP numerical error,
  so it was not selected. An exact callback alone does not fix AC
  convergence.
- **Diagnostics.** [diagnose.py](diagnose.py) repeats the reference-start
  diagnostic with the 16 most recent QPs recorded and exports the last one.
  The regression fixtures in `crates/clarabel/tests/data` came from such
  traces.

Reliable first-order convergence from the source and DC starts remains open,
as do other topologies and larger networks.
