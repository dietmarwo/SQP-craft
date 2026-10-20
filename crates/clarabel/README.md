# sqp-craft-clarabel

The Clarabel QP backend for the SQP-craft QP seam, the default backend of
[`sqp-craft-solver`](https://crates.io/crates/sqp-craft-solver), with optional
reuse of allocations and symbolic structure between QPs.

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft-clarabel --locked
```

MIT license.
