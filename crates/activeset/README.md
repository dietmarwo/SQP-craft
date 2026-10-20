# sqp-craft-activeset

A dense, warm-startable dual active-set (Goldfarb–Idnani) QP backend for the
SQP-craft QP seam, an alternative to the Clarabel backend of
[`sqp-craft-solver`](https://crates.io/crates/sqp-craft-solver).

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft-activeset --locked
```

MIT license.
