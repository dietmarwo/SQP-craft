# SQP5: Earth-Mars rendezvous with pykep-core

This executable chapter uses the existing native Rust solver through its Python API. The numerical definitions live with this chapter's owner; the shared driver supplies orchestration and figures.

## Model and decision

A fixed 300-day leg departs at MJD2000 9800. Ephemerides are converted from SI to AU-based canonical units. Optimize 20 throttle vectors, arrival mass and bounded departure excess velocity; study uses 40 segments. pykep supplies mismatch values and analytic Jacobians.

## Work through the lesson

The fixed 300-day leg departs at MJD2000 9800. Built-in Earth/Mars ephemerides freeze its endpoints. There are $3N$ throttle components plus normalized arrival mass, seven mismatch equalities and $N$ throttle-norm upper bounds.

1. Form a Lambert-informed start and check pykep-core's analytic mismatch/throttle Jacobians in nondimensional units. The Lambert impulse is context, not a low-thrust bound.
2. Keep `original_coast_model`: allowing zero throttle reaches an iteration limit near the nonsmooth coast region. A failed first-order solve is not evidence that rendezvous is physically impossible.
3. Declare a separate model with $\|u_i\|\ge0.02$. Smoke has 20 segments and converges to normalized arrival mass about 0.900114 for an initial 1,000 kg. It does not solve the original coast-permitting formulation by assertion.
4. Replay each impulsive midpoint segment with independent two-body DOP853 and the mass update; inspect mismatch in km and m/s. The native fixed Sims–Flanagan leg uses Lagrangian Kepler drifts with analytic STM Jacobians. Study uses 40 segments.

## Run

Install the pinned solver distributions and scientific dependencies using the [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md). Keep the patched repositories as sibling directories. From this repository's root:

```bash
python ../optimization-jobs/tutorials/structured/run.py --chapter SQP5 --output /absolute/new/sqp5
```

For the owner entry point, run `python tutorials/low-thrust-pykep/run.py --preset smoke --output /absolute/new/sqp5-direct`. It produces JSON; the shared driver additionally renders PNG/SVG figures. `--preset study` selects the declared larger cases. Output may be new or empty, and an occupied directory is refused.

## Independent answer and validation

A Lambert-informed start is declared. The model permitting exact coasts can stall near zero throttle; its full outcome is retained. A separate model requires throttle norm >=0.02, converges and is checked by an independent forward/backward midpoint-impulse replay using SciPy DOP853.

Every run retains native status, points and available diagnostics. LP candidates are checked against row/bound/integrality requirements; SQP results retain derivative checks and independently reconstructed KKT residuals. Application-specific references are computed separately. All failures remain in the report.

## Read the result

Read `summary.json` before interpreting `figure.png`. The report records the actual result; expected reference values are not substituted for missing numerical outcomes. A completed command job means that the chapter's executable checks completed, while individual solver attempts can still be nonconvergent where the lesson explicitly retains them.

The throttle-floor result is not a solution of an identical coast-permitting formulation. Report nonsmoothness and actual convergence. This example does not establish GTOC/SNOPT application readiness.

## Agent prompt

> Find a low-thrust Earth–Mars rendezvous for the fixed epoch and flight time using pykep-core derivatives. Retain the original coast-model outcome and treat a minimum-throttle-norm alternative as a distinct model. Replay each segment and report physical mismatch units.

Use the complete [prompt](prompt.md) for the workflow and evidence requirements.

## Follow-up exercise

Compare the two declared models and the extra fuel cost of excluding exact coasts. Inspect physical mismatch units and the forward/backward cut.

## Provenance and limits

The original plan's probes are historical planning evidence. The checked-in `results/smoke/summary.json` and figures record this implementation's executed case. Source and installed-package identities are also recorded by the shared driver. These are reproducible teaching experiments, not industrial or hardware performance certifications. Cross-formulation and cross-solver tables are descriptive.

## Recorded example

![Recorded SQP5 smoke result](results/smoke/figure.png)

Inspect the [complete smoke summary](results/smoke/summary.json) alongside the figure.

The [study summary](results/study/summary.json) and [study figure](results/study/figure.png) retain the declared study preset separately.
