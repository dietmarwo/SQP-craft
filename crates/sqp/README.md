# sqp-craft-solver

[Optimization ecosystem](https://dietmarwo.github.io/optimization-jobs/) ·
[Shared solver-selection and agent guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)

The native SQP driver for the [`sqp-craft`](https://crates.io/crates/sqp-craft)
contracts: damped BFGS, an L1 merit line search, feasibility restoration,
elastic mode, second-order correction and batch solves of many starts. QP
backends: [`sqp-craft-clarabel`](https://crates.io/crates/sqp-craft-clarabel)
(default) and [`sqp-craft-activeset`](https://crates.io/crates/sqp-craft-activeset).

```sh
cargo add sqp-craft sqp-craft-solver sqp-craft-clarabel
```

```rust
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, Result, SqpSolver,
    Status,
};
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::NativeSqp;

/// minimize (x0 - 1)^2 + (x1 - 2)^2  subject to  x0 + x1 <= 2,  0 <= x <= 4
struct Toy;

impl NlpProblem for Toy {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x[0] + x[1];
        Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }

    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
        g.copy_from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]);
        jac.copy_from_slice(&[1.0, 1.0]); // the dense 1 x 2 Jacobian pattern
        Ok(())
    }
}

fn main() -> Result<()> {
    let spec = ProblemSpec::new(
        Bounds::new(vec![0.0, 0.0], vec![4.0, 4.0]),
        Bounds::new(vec![f64::NEG_INFINITY], vec![2.0]),
        CsrPattern::dense(1, 2),
    )?;
    let result = NativeSqp::<ClarabelQp>::default().solve(
        &spec,
        &mut Toy,
        &[0.2, 0.4],
        &Options::default(),
        &CancellationToken::default(),
        None,
    )?;
    assert_eq!(result.status, Status::Converged); // a first-order KKT point
    println!("{:?}", result.candidate); // ≈ [0.5, 1.5]
    Ok(())
}
```

Limits: smooth objective and constraints with first derivatives; at most 256
variables (dense BFGS); a local method, so `Converged` is a first-order KKT
point within the tolerances, not a global optimum (use several starts for
multimodal models). For larger sparse problems an IPOPT-family solver fits
better. See the [API documentation](https://docs.rs/sqp-craft-solver) and the
[AI agent guide](https://github.com/dietmarwo/SQP-craft/blob/main/ai-context.md).

Part of [SQP-craft](https://github.com/dietmarwo/SQP-craft), which also holds the
Python package and the job worker. To build from source:

```sh
git clone https://github.com/dietmarwo/SQP-craft
cd SQP-craft
cargo test -p sqp-craft-solver --locked
```

MIT license.
