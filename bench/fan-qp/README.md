# Fan QP mapping and generic baseline

This is an eligibility and correctness experiment for the
[FanCyqlone interface](https://github.com/kul-optec/fancyqlone/tree/06f4bcfbab56d0fa9bbd82ad871929a3f0a464b1).
It adds no automatic QP backend. FanCyqlone remains an optional external
LGPL-3.0-or-later research comparator.

The fixture has two states, one input, eight stages, one or two branches,
and only the first input shared. Initial states are eliminated: the sparse
QP has 24 or 47 variables. Branch weights sum to one, so the shared input
cost is counted once. Exported FanOCP arrays include original fixed initial
states, stage/terminal costs, constraints and offsets. The generic QP keeps
an explicit inverse variable map and the objective constant.

```bash
cargo build --release --locked -p sqp-craft-clarabel --example fan_bridge
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/fan-qp/run.py \
  --binary target/release/examples/fan_bridge --output /absolute/new-fan-run
```

The Rust executable calls the existing `QpModel`/`ClarabelQp` seam. Each
returned candidate is checked in Python against original stage dynamics,
shared-input ownership, costs, bounds and KKT equations. Unconstrained
cases use a separate dense equality KKT oracle; bounded cases use SciPy
SLSQP as a reference. Unsupported coupling, cones, unequal dimensions and
nonpositive curvature are refused.

All 12 generic cases passed: two branch counts, three initial states and
bounded/unbounded-input variants. Maximum independently reconstructed
stationarity was about 1.16e-13. The external Conan dependency build reached
its 1,000-second limit while compiling batmat. Consequently **no FanCyqlone
solve or speed comparison was completed**, and the published spring-mass
benchmark and controlled multicore study remain unexecuted.

Tiny generic timings are descriptive. They do not support a specialized
backend or multicore speedup claim.
