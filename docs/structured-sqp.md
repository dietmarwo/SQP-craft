# Structured SQP with sparse curvature

This experimental feature targets smooth problems with bounded-size
overlapping elements. Full BFGS remains the default, with the existing
256-variable limit for full and uniform-block modes. For a structured solve
above that limit, use Clarabel. The native active-set SQP and its elastic
mode remain limited to the dense regime.

## Partitioned BFGS

In Python use `SqpOptions(hessian="partitioned", qp_reuse_workspace=True)`.
In Rust set `SqpOptions::hessian` to `Hessian::Partitioned`. Supply a fixed
CSR Jacobian and declare the nonlinear objective support. Use
`objective_pattern=[]` in Python or `with_objective_pattern(&[])` in Rust
for an affine objective. Without a declaration, small models use one dense
objective element. Above the dense limit the objective declaration is required.

Each optional `elements` group names constraint rows. The groups must
partition all rows, including any linear rows. Each nonlinear group's
variables are the union of its bounded, nonlinear rows' Jacobian columns.
The automatic grouping merges a row into an existing element when its
columns are a subset of that element. For multiple shooting, one group per
arc gives explicit control over the graph. Arrange variables in arc order
so neighboring elements overlap at their common node.

On an accepted step, an element restricts the step to its variables and
uses the secant `−Σ λᵢ(Jᵢ(x⁺) − Jᵢ(x))`. A nonlinear objective element
uses the objective gradient difference. The sign follows the convention
`∇f − Jᵀλ − z`. Overlapping elements divide their starting diagonal by
the number of elements using that coordinate. Coordinates outside all
elements keep a diagonal regularizer.

Each BFGS update is damped or repaired locally and checked by Cholesky.
A failed element update resets that element. If the assembled model is
rejected and a dense fallback fits, the driver can switch to full BFGS.
The report identifies the switch. Clarabel checks the assembled Hessian
by sparse Cholesky above 256 variables. Feasibility restoration accumulates
its Gauss–Newton matrix into a sparse pattern. KKT acceptance is independent
of the curvature approximation and uses the original model coordinates.

## Budget and diagnostics

`structured_memory_limit_bytes` defaults to 64 MiB. The maximum selectable
budget is 128 MiB. Preflight charges element storage, Jacobian and QP data,
and predicted factor fill. Dense Hessian patterns above the dense limit
are rejected. This is an estimate for the modeled structures, excluding
the interpreter, the model implementation and process overhead.

| `report.structure` field | Interpretation |
| --- | --- |
| `hessian_nnz` | Stored lower-triangular Hessian entries, including zeros |
| `hessian_factor_nnz` | Natural-order Hessian Cholesky entries, including the diagonal |
| `kkt_factor_nnz` | Symbolic KKT factor estimate with bounded-row and box-bound duals eliminated first |
| `qp_factor_nnz` | Clarabel's reported QDLDL factor entries under its own ordering, excluding the diagonal |
| `estimated_memory_bytes` | Estimate checked against the configured structured budget |
| `elements` | Row groups and counts for successful, damped, modified, skipped and reset updates |
| `fell_back_to_full` | A rejected assembled model triggered a dense fallback |

The job worker accepts the partitioned mode under the existing
`sqp.solve/1` profile for registered application models. See
[registering a model](job-api/models.md#registering-a-model) for the
worker limits and payload fields.

## Exact Lagrangian Hessian

Set `hessian="exact"` in Python or `Hessian::Exact` in Rust. Provide a
lower CSR pattern with all diagonal entries via `hessian_indptr` and
`hessian_indices`, or `ProblemSpec::with_lagrangian_hessian` in Rust.
The optional model callback `lagrangian_hessian(x, lambda)` returns the
values of `∇²f(x) − Σ λᵢ∇²cᵢ(x)` in that order. Rust writes to the
provided buffer. The callback counts against the model-call budget. The
first QP starts with zero constraint multipliers. Later QPs use the latest
multipliers at the accepted point.

`check_lagrangian_hessian` compares the declared matrix to bound-aware
finite differences of the analytic Lagrangian gradient, including entries
outside the pattern. Test with nonzero constraint multipliers if the model
has nonlinear rows. When a QP backend rejects an exact Hessian as indefinite,
SQP adds a diagonal shift using row sums as a sufficient bound. It allows up
to two such retries per iterate and reports the count and largest shift in
QP coordinates. The shift can slow convergence on problems with strongly
indefinite full-space curvature.

The [SQP4 and SQP6 exact examples](../tutorials/_support/exact_hessian.py)
run both modes from the same start and recheck first-order residuals.
In the tested build the brachistochrone took 13 full-BFGS steps and
75 exact steps, with 76 diagonal shifts. The kinetics fit took 16 and 4
steps respectively, with zero shifts. Both models passed the Hessian
derivative check. The multiple-shooting leg still uses partitioned BFGS
because the trajectory core supplies first-order sensitivities.

## Measured SQP5 scaling

The [target card](target-card.md) defines the physical case, starts and
budgets. The independent verifier reconstructs KKT from fresh model
evaluations and uses DOP853 for trajectory replay. These measurements use
four throttle segments per arc and Clarabel workspace reuse.

| Variables | Accepted steps | Independent KKT | Hessian entries | Clarabel factor entries |
| ---: | ---: | --- | ---: | ---: |
| 130 | 42 | Pass | 2,050 | 3,687 |
| 244 | 47 | Pass | 3,988 | 7,101 |
| 491 | 75 | Pass | 8,187 | 14,498 |
| 966 | 108 | Pass | 16,262 | 28,723 |
| 1,916 | 127 | Pass | 32,412 | 57,173 |
| 4,994 | 212 | Pass, extended time budget | 84,738 | 149,351 |

On the same unscaled Lambert starts, full BFGS took 74 and 106 accepted
steps at 130 and 244 variables, against 42 and 47 for partitioned BFGS.
All four arrival masses agree with the SLSQP references to within 1e-6 kg.

Both offset-start checks passed at 966 and 1,916 variables. The 4,994-variable
case was feasible at 30 seconds but failed the stationarity threshold. It
passed in 31.5–32.9 seconds in two runs with a 90-second budget. The Python
process peaked at 111–113 MiB RSS, with a 26.3 MiB structured storage
estimate. Factor entries
grow approximately in proportion to arc count for this layout. Interior-point
work and the number of SQP steps still grow.

The same 966- and 1,916-variable Lambert starts failed the independent QP
check with `scaling="jacobian"`. The reported runs use `scaling="none"`,
which preserves the units in this already nondimensionalized model.

Run one case from the repository root with sqp-craft 0.18.0 or later and
pykep-rust 0.1.6 or later installed (or both built from source).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m bench.trajectory_scale \
  --arcs 101 --method partitioned --scaling none --output transfer-1916.json
```

The [SQP5 tutorial](../tutorials/low-thrust-pykep/README.md#multiple-shooting)
includes the derivative and experiment-craft commands. The raw records of
every run, the sealed campaign results and the comparator runs are in
[benchmarks/structured](benchmarks/structured/README.md).

## Other structured problems

The [structured S2MPJ selection](benchmarks/structured-selection.json)
(S2MPJ `fea6a700`) has three affine-objective problems with nonlinear sparse
constraints above 256 variables. The benchmark harness's independent
checker judged each result; Uno 2.9.3 ran through unopy 0.4.17, all with a
500-step limit.

| Case | Variables | Partitioned SQP | Uno filterSQP | Uno IPOPT preset |
| --- | ---: | --- | --- | --- |
| HANGING 10 × 10 | 300 | Pass, 19 steps | Pass, 89 | Pass, 149 |
| CATENARY 166 | 501 | Pass, 65 steps | Step limit | Step limit |
| UBH5 40 | 410 | Step limit | Step limit | Step limit |

The large-problem route remains experimental, especially where the QP is
ill-conditioned or the objective and constraints do not split into small
cliques.
