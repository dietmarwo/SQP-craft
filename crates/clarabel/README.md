# sqp-craft-clarabel

[Optimization ecosystem](https://dietmarwo.github.io/optimization-jobs/) ·
[Shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)

The Clarabel QP backend for the SQP-craft QP seam, the default backend of
[`sqp-craft-solver`](https://crates.io/crates/sqp-craft-solver), with optional
reuse of allocations and symbolic structure between QPs.

Dense Hessians are limited to 256 variables. Above that, the adapter takes
sparse Hessians from [structured SQP](https://github.com/dietmarwo/SQP-craft/blob/main/docs/structured-sqp.md): it checks a symbolic memory
estimate and positive definiteness by sparse Cholesky before converting
the QP.

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft-clarabel --locked
```

MIT license.
