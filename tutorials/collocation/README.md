# SQP4: Optimal control by collocation and independent replay

This executable chapter uses the existing native Rust solver through its Python API. The numerical definitions live with this chapter's owner; the shared driver supplies orchestration and figures.

## Model and decision

Transcribe the brachistochrone with states x,y,v, control theta and free final time, then reuse the existing Bryson-Ho orbit-radius Python model without changing its equations. Both use trapezoidal defects and analytic sparse derivatives.

## Work through the lesson

For the brachistochrone, states $(x,y,v)$ use downward-positive $y$ and angle control $\theta$:

$$\dot x=v\sin\theta,\quad\dot y=v\cos\theta,\quad\dot v=g\cos\theta.$$

Time is scaled to $[0,1]$ and final time is a decision. Trapezoidal defects enforce adjacent-node dynamics; they do not guarantee the interpolated path satisfies the ODE between nodes.

1. Solve 10 and 20 intervals, then 40 in study. Compare with cycloid time 0.5828954632 s; the retained smoke times are about 0.5831980 and 0.5829773 s.
2. Interpolate control and replay with independent DOP853, retaining endpoint errors separately from defects. At zero initial speed, $\theta$ still affects acceleration through $g\cos\theta$.
3. Import the existing orbit-radius model unchanged. Its 20-interval final radius is about 1.52206048; study uses 40 and 60 intervals. Replay terminal radial velocity and circular-speed residuals. Mesh-dependent values are compared at the same mesh, not silently against a continuous optimum.

## Run

Install the pinned solver distributions and scientific dependencies using the [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md). Keep the patched repositories as sibling directories. From this repository's root:

```bash
python ../optimization-jobs/tutorials/structured/run.py --chapter SQP4 --output /absolute/new/sqp4
```

For the owner entry point, run `python tutorials/collocation/run.py --preset smoke --output /absolute/new/sqp4-direct`. It produces JSON; the shared driver additionally renders PNG/SVG figures. `--preset study` selects the declared larger cases. Output may be new or empty, and an occupied directory is refused.

## Independent answer and validation

Cycloid descent time is 0.5828954632 seconds. Smoke solves 10/20 bead intervals and the 20-interval orbit; study reaches bead 40 and orbit 40/60. Independent SciPy DOP853 integration replays each control and reports endpoint differences separately from defects.

Every run retains native status, points and available diagnostics. LP candidates are checked against row/bound/integrality requirements; SQP results retain derivative checks and independently reconstructed KKT residuals. Application-specific references are computed separately. All failures remain in the report.

## Read the result

Read `summary.json` before interpreting `figure.png`. The report records the actual result; expected reference values are not substituted for missing numerical outcomes. A completed command job means that the chapter's executable checks completed, while individual solver attempts can still be nonconvergent where the lesson explicitly retains them.

The initial bead control affects acceleration even when speed is zero. Mesh feasibility is not continuous-trajectory accuracy; the orbit remains a development example, not the unavailable application target.

## Agent prompt

> Transcribe the brachistochrone and supplied orbit-radius transfer by collocation. Refine the meshes, compare like-for-like references, and independently integrate the controls. Keep defects and physical replay errors separate.

Use the complete [prompt](prompt.md) for the workflow and evidence requirements.

## Follow-up exercise

Omit the free-time factor or flip a gravity sign in a test copy and check the replay. Compare refinement with merely tightening solver tolerances.

## Provenance and limits

The original plan's probes are historical planning evidence. The checked-in `results/smoke/summary.json` and figures record this implementation's executed case. Source and installed-package identities are also recorded by the shared driver. These are reproducible teaching experiments, not industrial or hardware performance certifications. Cross-formulation and cross-solver tables are descriptive.

## Recorded example

![Recorded SQP4 smoke result](results/smoke/figure.png)

Inspect the [complete smoke summary](results/smoke/summary.json) alongside the figure.

The [study summary](results/study/summary.json) and [study figure](results/study/figure.png) retain the declared study preset separately.
