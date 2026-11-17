# EPIGRIDS: checked DC dispatch and an AC regression study

This experiment uses the 250-bus New England MATPOWER data from
[EPIGRIDS v1.0.0](https://github.com/WISPO-POP/SyntheticElectricNetworkModels/tree/v1.0.0),
CC BY 4.0, by Jonathan Snodgrass and collaborators. Original case bytes are
retained. The three easy cases are operating points on one topology.

The constrained literal parser never evaluates MATLAB. An independent
matpowercaseframes loader reads the original file. The sparse AC model has
583 variables with 42 active generators; an offline generator changes this
dimension. Both ends of each branch use apparent-power limits. A zero/zero
angle-limit pair follows MATPOWER's unconstrained convention.

The DC LP uses 16 secant intervals per quadratic cost and a separate tangent
lower model. LP-IP-craft and SciPy HiGHS solve identical rows. The original
quadratic costs, the approximation bound and the model difference from AC
are reported separately.

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 pypower==5.1.21 matpowercaseframes==2.1.1
# Build/install this repository and sibling LP-IP-craft first.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/epigrids/test_power.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/epigrids/run.py \
  --data /absolute/SyntheticElectricNetworkModels --output /absolute/new-grid-run
```

The November study matched HiGHS on all eight feasible LPs across four
operating points. Both solvers rejected the base-case DC LP: minimum
generation exceeds its load. No native AC attempt met the frozen acceptance
criteria. Clarabel rejected step QPs, usually with `AlmostSolved`; the
stress fixture also reached a restoration limit. These are numerical
regression observations, with no AC infeasibility certificate or accepted
native AC dispatch.

The independent AC reference is PYPOWER's interior-point solver. Its
voltage/admittance reconstruction checks final native points independently
of the model callback. Original-currency stationarity/complementarity must
be at most 1e-6; balances and normalized limits at most 1e-7. Objective
normalization by 10,000 is undone before that stationarity gate. Tolerances
were retained on failure.

[diagnose.py](diagnose.py) repeats the reference-start diagnostic with 16
most recent recorded QPs and exports the last one. This is separate from the frozen
run matrix. The full execution log and raw evidence accompany the delivery.

The November 17 follow-up adds bounded QP accuracy recovery in the native
Clarabel adapter. Its selected configuration uses partitioned curvature, an
initial Hessian scale of 0.01, the existing augmented-Lagrangian merit and a
tighter internal QP tolerance of 1e-12. Selection used the primary case only;
the final checker and the 60-second / 300-step / 5,000-callback caps are
unchanged. The original defaults remain available. Reproduce the follow-up:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/epigrids/run.py \
  --data /absolute/SyntheticElectricNetworkModels --output /absolute/new-ac-run \
  --merit augmented-lagrangian --qp-tolerance 1e-12 --initial-hessian-scale 0.01
```

Each output records its actual solver options and separate fresh final KKT
residuals. The follow-up accepted 5/14 executed AC runs (5/9 on the easy
operating points): all three Winter68Generous starts and the reference
diagnostic starts on Winter12Generous and Fall116Generous. A sixth native
first-order point on Spring13Tight passed the physical/KKT checks, but the
original PYPOWER reference failed, so it remains outside the frozen full
acceptance count. Other ordinary starts remain regression targets.

The optional `--hessian exact` arm uses sparse analytic PYPOWER
second derivatives, checked against directional gradient differences; it is
experimental and was not selected. Its first reference-start probe failed
numerically, so an exact callback alone is not an AC convergence fix.
Larger-network studies remain future work.
