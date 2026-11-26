# Shared-root spring–mass QP study

The published case is the first horizon and first seed in FanCyqlone Figure 3
(`wang-boyd-2008`, 12 masses, 32 transitions, four branches, seed 0). The
fixtures share one six-component first input. The reduced QP has 3,822
variables, 3,072 equalities and 2,286 finite two-sided boxes. The fixed
initial state is substituted and its objective contribution is retained.
The CSR transport uses 92,928 dynamics entries and 3,822 Hessian entries.

The upstream reference uses [FanCyqlone at `06f4bcf`](https://github.com/kul-optec/fancyqlone/tree/06f4bcfbab56d0fa9bbd82ad871929a3f0a464b1),
built independently with the fixture-export instrumentation from the
experiment package. It uses the same case generator as the upstream
benchmark. The instrumented executable records evaluated discretization
coefficients, reconstructed primal–dual vectors, solver status and selected
linear-solver buffer sizes. FanCyQPALM and the existing Clarabel adapter
are compared in the original stage coordinates.

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
`--fixture /path/to/spring-mass.fixture --reference-solution /path/to/fancyqpalm.json`.
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
