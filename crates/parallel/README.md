# sqp-craft-parallel

[Optimization ecosystem](https://dietmarwo.github.io/optimization-jobs/) ·
[Shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)

Explicit, bounded native execution resources for SQP-craft: Rayon pools of a
declared size for the large QP kernels and native models. No global Rayon
pool is configured.

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft-parallel --locked
```

MIT license.
