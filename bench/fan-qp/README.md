# Shared-root spring–mass QP study

The published case is the first horizon and first seed in FanCyqlone Figure 3
(`wang-boyd-2008`, 12 masses, 32 transitions, four branches, seed 0). The
fixtures share one six-component first input. The reduced QP has 3,822
variables, 3,072 equalities and 2,286 finite two-sided boxes. The fixed
initial state is substituted and its objective contribution is retained.
The CSR transport uses 92,928 dynamics entries and 3,822 Hessian entries.

The upstream reference uses [FanCyqlone at `06f4bcf`](https://github.com/kul-optec/fancyqlone/tree/06f4bcfbab56d0fa9bbd82ad871929a3f0a464b1),
built independently with a fixture-export instrumentation. It uses the same
case generator as the upstream benchmark, whose `benchmark-scaling` command
runs Figure 3 with `-M 12 -J 4`, horizons from 32 and seeds from 0, and the
benchmark's solver settings (tolerances 1e-8, initial penalty 20, zero
start). It differs from the benchmark in using scalar kernels and serial
branches. The instrumented executable records evaluated discretization
coefficients, reconstructed primal–dual vectors, solver status and selected
linear-solver buffer sizes. FanCyQPALM and the existing Clarabel adapter
are compared in the original stage coordinates.

The instrumentation patch and its build script are kept out of this MIT-licensed repository:
they modify LGPL-3.0 sources. They are in the delivery archive
`SQP-craft-FanCyqlone-patch.zip` (SHA-256
`86c3951b12fb6aafd1d1e375e4b0d08f5f570c04adc23940c7367a84a0e22968`) under
`reference/`. What the executable exported is committed here, so everything
except the FanCyQPALM solve itself replays without it.

## Results

[results/2026-11-26](results/2026-11-26) holds the exported fixture, the
FanCyQPALM solution, the residual and fill records of the replay below, the
toolchain of the reference build and the [delivered log](results/2026-11-26/execution-log.md).

| Measurement | Clarabel through SQP-craft | FanCyQPALM, original stage rows | First exact SQP step |
|---|---:|---:|---:|
| Recomputed objective | 1778.031254933646 | 1778.031254158380 | 1778.031254933746 |
| Dynamics residual | 1.71e-13 | 1.78e-15 | 3.78e-14 |
| Box violation | 1.30e-14 | 3.73e-9 | 0 |
| Stationarity | 1.23e-13 | 1.79e-13 | 1.12e-12 |
| Complementarity | 2.25e-10 | 2.30e-8 | 4.40e-10 |

All values are infinity norms in original units. The limits are 1e-8, and
1e-6 for complementarity. The two objectives differ by 4.36e-10 relative.

| Ordering of the box-reduced KKT graph (6,894 vertices) | Factor entries incl. diagonal | Largest front |
|---|---:|---:|
| Minimum degree | 204,333 | 48 |
| Minimum degree, shared input deferred | 204,333 | 48 |
| Reverse-stage order, shared input last | 240,081 | 54 |
| Reverse-stage order, shared input first | 243,522 | 97 |

The equality of the first two rows does not depend on the tie-break of the
minimum-degree rule: index order, reverse index order and three random orders
all give 204,333 for both. Unrestricted minimum degree eliminates the shared
input late on its own (positions 6,144 to 6,149 of 6,894 with the index
tie-break).

This is one convex instance on one core. It establishes that the mapping and
the checks are right and how much storage the sparse form needs. It says
nothing about speed.

Replay from the committed inputs, without FanCyqlone:

```sh
r=bench/fan-qp/results/2026-11-26
cargo build --release --locked -p sqp-craft-clarabel --example fan_bridge
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/fan-qp/run.py \
  --binary target/release/examples/fan_bridge --fixture $r/spring-mass.fixture.gz \
  --reference-solution $r/fancyqpalm.json.gz --exact-step --output fan-replay
```

On two x86-64 hosts (AMD EPYC, Intel Xeon) and with Rust 1.97 and 1.98 this
gave the same bytes for the sparse input, Clarabel's solution vectors, the
residual record and the fill record. `tests/test_fan_qp.py` checks the
committed inputs and records; with `FAN_BRIDGE` set to the built example it
also runs the toy cases and this replay, as CI does.

## Run

From the SQP-craft root, build the Rust example. Run the 12 toy regressions,
including two native upstream storage round trips when the comparator is
supplied. Each output directory must be new.

```sh
cargo build --release --locked -p sqp-craft-clarabel --example fan_bridge
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/fan-qp/run.py \
  --binary target/release/examples/fan_bridge --output fan-toys
```

For the published comparison, install NumPy, SciPy and the SQP-craft Python
package built from this checkout. Pass the instrumented upstream executable
built using the accompanying reference patch. Use `--exact-step` to record
the QP from the normal exact-Hessian SQP callback.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  python bench/fan-qp/run.py --binary target/release/examples/fan_bridge \
  --upstream /path/to/spring-mass-fixture --published --exact-step \
  --output fan-published
```

To replay the comparison offline, replace `--upstream … --published` with
`--fixture /path/to/spring-mass.fixture --reference-solution /path/to/fancyqpalm.json`
(plain or gzipped).
Supply `--upstream` with the toy command to check the C++ layout for a nonzero
initial state with finite and unbounded sides. A missing reference in
`--fixture` mode still produces the generic certificate and fill records and
returns a nonzero exit status.

## Files and checks

- The stage interchange starts with `fan-ocp/1 J N nx nu ny ny_terminal`.
  For each branch, write the fixed initial state followed by the stages in
  order. Ordinary stages hold row-major `[A B]`, the affine offset,
  the symmetric stage Hessian, the gradient, box-selector rows and lower
  and upper sides. The terminal stage omits dynamics and inputs.
- The bridge input starts with `fan-qp/1 n m nnz_H nnz_E`. For each of
  the lower Hessian and the dynamics Jacobian, send CSR row pointers,
  column indices and values. Append the gradient, variable bounds and
  row bounds as vectors. Bounds use `-inf` and `inf` for open sides.
- `residuals.json` contains absolute KKT checks from the stage data, in
  original units. The tests cover root-copy consistency, the dynamics and
  box multiplier signs, off-branch support and the objective constant.
- `fill.json` uses the same box-reduced KKT graph for symbolic orderings.
  It includes the diagonal, records the permutations and distinguishes
  the generic count from Clarabel's QDLDL diagnostics. The native upstream
  selected storage inventory is in `fancyqpalm.json`.
- `allocations.json` gives CSR shapes and bytes, variable and bound maps,
  and assembly-only checks at double the horizon and double the branches.

Both solvers passed the original-coordinate checks for the Figure 3 case.
Minimum degree gave the same symbolic count with the shared input deferred.
The fixed reverse-stage ordering increased symbolic entries by roughly 17.5%.
The two QP engines use different numeric systems. Timing comparisons require
the controlled four- and eight-core study described in the investigation plan.
