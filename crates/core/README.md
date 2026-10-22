# sqp-craft

[Optimization ecosystem](https://dietmarwo.github.io/optimization-jobs/) ·
[Shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)

Problem, callback, solver and result contracts for SQP on smooth constrained
problems with sparse (CSR) constraint Jacobians (`ProblemSpec`, `NlpProblem`,
`SqpSolver`, `SolveResult`), KKT residuals, the QP seam for backends and a
finite-difference derivative checker. No dependencies; the solver is
[`sqp-craft-solver`](https://crates.io/crates/sqp-craft-solver), whose README
has a complete example:

```sh
cargo add sqp-craft sqp-craft-solver sqp-craft-clarabel
```

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft --locked
```

MIT license.
