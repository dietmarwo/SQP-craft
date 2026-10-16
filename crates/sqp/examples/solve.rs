//! Runs native SQP on a toy problem and checks KKT residuals for a
//! hand-computed solution: `cargo run -p sqp-craft-solver --example solve`.
//!
//! minimize (x0 - 1)^2 + (x1 - 2)^2  subject to  x0 + x1 <= 2,  0 <= x <= 4

use sqp_craft::kkt::nlp_residuals;
use sqp_craft::{
    Bounds, CancellationToken, Control, CsrPattern, Multipliers, NlpProblem, Options, ProblemSpec,
    Progress, Result, SqpSolver,
};
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::NativeSqp;

struct Toy;

impl NlpProblem for Toy {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        c[0] = x[0] + x[1];
        Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }

    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
        jac.fill(1.0);
        Ok(())
    }
}

fn main() -> Result<()> {
    let spec = ProblemSpec::new(
        Bounds::new(vec![0.0, 0.0], vec![4.0, 4.0]),
        Bounds::new(vec![f64::NEG_INFINITY], vec![2.0]),
        CsrPattern::dense(1, 2),
    )?;
    let mut print = |p: &Progress| -> Result<Control> {
        println!("observer: {p:?}");
        Ok(Control::Continue)
    };

    // Start away from the solution x* = (0.5, 1.5).
    let result = NativeSqp::<ClarabelQp>::default().solve(
        &spec,
        &mut Toy,
        &[0.2, 0.4],
        &Options::default(),
        &CancellationToken::default(),
        Some(&mut print),
    )?;
    println!("status: {:?} ({})", result.status, result.message);

    // grad f(x*) = (-1, -1) = J^T λ with λ = -1: the active upper bound gets a negative multiplier.
    assert_eq!(result.status, sqp_craft::Status::Converged);
    let evaluation = result.evaluation.expect("start point was evaluated");
    let multipliers = Multipliers {
        constraints: vec![-1.0],
        variables: vec![0.0, 0.0],
    };
    let r = nlp_residuals(&spec, &[0.5, 1.5], &evaluation, &multipliers)?;
    println!("KKT residuals with hand-computed multipliers: {r:?}");
    assert!(r.max() < 1e-6);
    Ok(())
}
