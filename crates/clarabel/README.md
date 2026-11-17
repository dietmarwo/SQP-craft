# sqp-craft-clarabel

[Optimization ecosystem](https://dietmarwo.github.io/optimization-jobs/) ·
[Shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)

The Clarabel QP backend for the SQP-craft QP seam, the automatic structured backend of
[`sqp-craft-solver`](https://crates.io/crates/sqp-craft-solver), with optional
reuse of allocations and symbolic structure between QPs.

Dense Hessians are limited to 256 variables. Above that, the adapter takes
sparse Hessians from [structured SQP](https://github.com/dietmarwo/SQP-craft/blob/main/docs/structured-sqp.md): it checks a symbolic memory
estimate and positive definiteness by sparse Cholesky before converting
the QP.

Every solution, including an `AlmostSolved` candidate, must pass the shared
original-coordinate KKT check. The check also covers the step rounded onto
its variable bounds, as used by SQP. A rejected cold candidate can receive
sparse equality-KKT polishing: active variable bounds are eliminated,
wrong-sign active guesses can be released, and iterative refinement uses
the unregularized equations. All inactive rows, bounds and dual signs are
still independently checked. This private recovery is bounded to three
active sets and eight refinements per factor, with a conservative allocation
cap; it drops the engine workspace before allocating its factor. A rejected
equilibrated cold solve can also retry without equilibration at tighter
internal accuracy. All work shares the call's time and iteration budget;
the independent acceptance tolerance never changes. Polished solves retain
the conversion cache but rebuild the engine workspace on the next call.

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft-clarabel --locked
```

MIT license.
