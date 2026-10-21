# sqp-craft-solver

[Optimization ecosystem](https://dietmarwo.github.io/optimization-jobs/) ·
[Shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)

The native SQP driver for the [`sqp-craft`](https://crates.io/crates/sqp-craft)
contracts: damped BFGS, an L1 merit line search, feasibility restoration,
elastic mode, second-order correction and batch solves of many starts. QP
backends: [`sqp-craft-clarabel`](https://crates.io/crates/sqp-craft-clarabel)
(default) and [`sqp-craft-activeset`](https://crates.io/crates/sqp-craft-activeset).

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft-solver --locked
```

MIT license.
