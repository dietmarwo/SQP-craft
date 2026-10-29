# SQP3: Hanging chain: separate mesh and solver error

This executable chapter uses the existing native Rust solver through its Python API. The numerical definitions live with this chapter's owner; the shared driver supplies orchestration and figures.

## Model and decision

Uniform rigid links span one unit with total length two. Minimize gravitational potential subject to squared link-length equalities. The interior coordinates give 2(N-1) variables and a sparse banded Jacobian.

## Work through the lesson

For $N$ uniform rigid links of total length $L=2$ spanning $D=1$, optimize the $2(N-1)$ interior coordinates:

$$\min\sum_{i=0}^{N-1}\frac{L}{2N}(y_i+y_{i+1}),\qquad
(x_{i+1}-x_i)^2+(y_{i+1}-y_i)^2=(L/N)^2.$$

1. Check the banded constraint Jacobian and solve smoke meshes 16 and 32. Link lengths and KKT residuals assess the discretized optimization.
2. Independently solve one scalar horizontal-tension equation for the exact discrete equilibrium. Objective disagreement with that reference is solver error.
3. Compare the discrete reference with the continuum catenary: $a=0.229640215078$, potential approximately -0.911208463318. This difference is discretization error. Study continues through 64 and 128 links; compare the actual errors before claiming an observed order.
4. The implementation records the hard dimension boundary explicitly: 129 links use 256 variables; 128 is the largest admissible power-of-two mesh. The reflected arch and additional warm-start experiments remain follow-up exercises.

## Run

Install the pinned solver distributions and scientific dependencies using the [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md). Keep the patched repositories as sibling directories. From this repository's root:

```bash
python ../optimization-jobs/tutorials/structured/run.py --chapter SQP3 --output /absolute/new/sqp3
```

For the owner entry point, run `python tutorials/hanging-chain/run.py --preset smoke --output /absolute/new/sqp3-direct`. It produces JSON; the shared driver additionally renders PNG/SVG figures. `--preset study` selects the declared larger cases. Output may be new or empty, and an occupied directory is refused.

## Independent answer and validation

An independent one-dimensional tension solve gives exact discrete equilibrium; the catenary gives the continuum reference. Smoke runs 16/32 links; study adds 64/128. Their potential differences separate numerical convergence from discretization error.

Every run retains native status, points and available diagnostics. LP candidates are checked against row/bound/integrality requirements; SQP results retain derivative checks and independently reconstructed KKT residuals. Application-specific references are computed separately. All failures remain in the report.

## Read the result

Read `summary.json` before interpreting `figure.png`. The report records the actual result; expected reference values are not substituted for missing numerical outcomes. A completed command job means that the chapter's executable checks completed, while individual solver attempts can still be nonconvergent where the lesson explicitly retains them.

An infeasible point can have an objective below the true minimum. Continuation interpolates a declared start and does not resume solver state.

## Agent prompt

> Solve the hanging chain on the declared meshes. Use the exact discrete equilibrium and continuum catenary to separate solver error from discretization error. State the dimension limit without confusing 128 with 129 links.

Use the complete [prompt](prompt.md) for the workflow and evidence requirements.

## Follow-up exercise

Reflect a solution upward and inspect first-order conditions. Explain why 128 is the largest power-of-two mesh here, while the exact 256-variable limit permits 129 links.

## Provenance and limits

The original plan's probes are historical planning evidence. The checked-in `results/smoke/summary.json` and figures record this implementation's executed case. Source and installed-package identities are also recorded by the shared driver. These are reproducible teaching experiments, not industrial or hardware performance certifications. Cross-formulation and cross-solver tables are descriptive.

## Recorded example

![Recorded SQP3 smoke result](results/smoke/figure.png)

Inspect the [complete smoke summary](results/smoke/summary.json) alongside the figure.

The [study summary](results/study/summary.json) and [study figure](results/study/figure.png) retain the declared study preset separately.
