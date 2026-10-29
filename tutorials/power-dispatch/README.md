# X2: Power dispatch: DC LP versus AC physics

This chapter runs native Rust solvers through their Python APIs. Its model and independent checks live in this directory; the shared driver in optimization-jobs adds figures and command-job execution.

## Model and decision

A declared synthetic five-bus network has two generators. DC-OPF uses linear angle/flow equations. AC-OPF includes voltage magnitudes, active/reactive generation, nonlinear balance and squared apparent-power limits.

## Work through the lesson

DC-OPF linearizes voltage around unit magnitude and small phase differences. AC-OPF instead enforces complex power balance

$$P_i+\mathrm{i}Q_i=V_i\overline{(YV)_i}.$$

The frozen five-bus data include branch impedances and limits, active/reactive demand, voltage bounds and generator costs.

1. Solve the DC LP and recompute its bounds, balance rows and line limits from the matrix. Then replay that active dispatch in an independent AC power flow: the second generator keeps its DC output with zero reactive output, and the slack generator absorbs the losses. `dc_ac_replay.limits` reports the slack output, voltages, branch flows and losses against the AC limits; a frozen DC dispatch need not satisfy them.
2. Solve the smooth AC problem and reconstruct KKT. Recorded cost is about 2.232961 versus DC's 2.2; these are different models, so the difference is not an optimizer ranking.
3. Independently rebuild the complex admittance calculation and recompute all balance rows. The retained maximum balance mismatch is about $3.6\times10^{-12}$ in the model's per-unit system.
4. A converged AC solution is a checked first-order point. Global certification would require additional methods; the DC optimum does not certify the nonconvex AC optimum.

## Run

Install the pinned solver packages and scientific dependencies as described in the [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md), with the repositories it lists cloned side by side. From this repository's root:

```bash
python ../optimization-jobs/tutorials/structured/run.py --chapter X2 --output /absolute/new/x2
```

To run the chapter on its own, use `python tutorials/power-dispatch/run.py --preset smoke --output /absolute/new/x2-direct`. It produces JSON; the shared driver additionally renders PNG/SVG figures. `--preset study` runs the larger declared cases; a chapter without them repeats the smoke case. Output may be new or empty, and an occupied directory is refused.

## Independent answer and validation

The DC candidate is checked against its own rows and bounds, and its dispatch is replayed in an independent AC power flow (SciPy `root`, Powell hybrid method). Native SQP solves AC-OPF with complex-step derivatives of the same balance equations. Their finite-difference check confirms that the complex step is valid for that code, not that the equations are right; the separate complex-arithmetic power-balance check verifies the final point.

Every run keeps the native status, points and available diagnostics. SQP results keep their derivative checks and independently reconstructed KKT residuals. Application-specific references are computed separately. All failures remain in the report.

## Read the result

Read `summary.json` before interpreting `figure.png`. The report records the actual result; expected reference values are not substituted for missing numerical outcomes. A completed command job means that the chapter's executable checks completed, while individual solver attempts can still be nonconvergent where the lesson explicitly retains them.

DC and AC are different physical models. AC convergence is first-order numerical evidence; neither model supplies a general global certificate for nonlinear power flow.

## Agent prompt

> Find a low-cost power dispatch for the supplied five-bus network. Replay the DC decision in AC physics, then solve the AC problem and check complex power balance independently. Explain the local and global claim boundary.

Use the complete [prompt](prompt.md) for the workflow and evidence requirements.

## Follow-up exercise

Reduce a line rating or increase demand. Inspect AC voltage/reactive requirements and why the slack must absorb losses.

## Provenance and limits

The checked-in `results/` record one run of each preset. The shared driver writes source and package identities to `environment.json` for the runs it makes. These are reproducible teaching experiments, not industrial or hardware performance certifications. Tables that compare formulations or solvers are descriptive.

## Recorded example

![Recorded X2 smoke result](results/smoke/figure.png)

Inspect the [complete smoke summary](results/smoke/summary.json) alongside the figure.

The [study summary](results/study/summary.json) and [study figure](results/study/figure.png) record the study preset.
