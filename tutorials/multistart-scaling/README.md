# SQP2: Multistart, units and QP backends

This executable chapter uses the existing native Rust solver through its Python API. The numerical definitions live with this chapter's owner; the shared driver supplies orchestration and figures.

## Model and decision

The simplex calibration minimizes sum(x_i^2) on four nonnegative shares summing to one. The diagnostic problem is (x^2-1)^2+(y-x)^2+0.2*x with y=x and [-2,2] bounds. Compare balanced and deliberately scaled presentations, with identical physical starts.

## Work through the lesson

The physical problem is

$$\min (x^2-1)^2+(y-x)^2+0.2x,\qquad y-x=0.$$

On the feasible line, stationary points solve $4x(x^2-1)+0.2=0$. Root isolation supplies the two minima and intervening maximum independently.

1. Run four physical starts through each backend and scaling choice. In smoke this gives 32 double-well solves; study also includes the `linear` scaling mode.
2. Change coordinates with $(x,y)=Du$ and multiply the equality by $R$. The physical feasible set is unchanged, but gradients, multipliers and stopping scales transform. Check stationarity after mapping back to physical units.
3. Read the separate simplex tests for active bounds. Multipliers at a bound are part of KKT; an unconstrained gradient norm is insufficient.
4. Set the tilt to zero and start at the origin. It is first order but is a maximum along the feasible line. The implementation retains this counterexample instead of interpreting convergence as a minimum certificate.

## Run

Install the pinned solver distributions and scientific dependencies using the [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md). Keep the patched repositories as sibling directories. From this repository's root:

```bash
python ../optimization-jobs/tutorials/structured/run.py --chapter SQP2 --output /absolute/new/sqp2
```

For the owner entry point, run `python tutorials/multistart-scaling/run.py --preset smoke --output /absolute/new/sqp2-direct`. It produces JSON; the shared driver additionally renders PNG/SVG figures. `--preset study` selects the declared larger cases. Output may be new or empty, and an occupied directory is refused.

## Independent answer and validation

Simplex value is 0.25. Cubic root bracketing identifies two minima and a stationary maximum in the double well. The smoke matrix has 32 double-well solves, four simplex solves and an alpha-zero stationary-point demonstration. Native and physical-unit KKT checks are both retained.

Every run retains native status, points and available diagnostics. LP candidates are checked against row/bound/integrality requirements; SQP results retain derivative checks and independently reconstructed KKT residuals. Application-specific references are computed separately. All failures remain in the report.

## Read the result

Read `summary.json` before interpreting `figure.png`. The report records the actual result; expected reference values are not substituted for missing numerical outcomes. A completed command job means that the chapter's executable checks completed, while individual solver attempts can still be nonconvergent where the lesson explicitly retains them.

Internal solver scaling is distinct from changing model units. Absolute tolerances change meaning across coordinates, so conclusions need the canonical physical check. Tiny models do not establish a backend speed ranking.

## Agent prompt

> Solve the tilted double well from multiple physical starts, then change coordinate units and equality-row scale. Map points and multipliers back to physical units and explain why a first-order point can still be a maximum.

Use the complete [prompt](prompt.md) for the workflow and evidence requirements.

## Follow-up exercise

Inspect the alpha-zero point (0,0): first-order conditions hold but it is not a minimum. Corrupt a multiplier or omit its coordinate transformation and check the result.

## Provenance and limits

The original plan's probes are historical planning evidence. The checked-in `results/smoke/summary.json` and figures record this implementation's executed case. Source and installed-package identities are also recorded by the shared driver. These are reproducible teaching experiments, not industrial or hardware performance certifications. Cross-formulation and cross-solver tables are descriptive.

## Recorded example

![Recorded SQP2 smoke result](results/smoke/figure.png)

Inspect the [complete smoke summary](results/smoke/summary.json) alongside the figure.

The [study summary](results/study/summary.json) and [study figure](results/study/figure.png) retain the declared study preset separately.
