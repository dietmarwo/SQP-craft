//! Shared toy model: minimize (x0 - 1)^2 + (x1 - 2)^2 subject to x0 + x1 = 3, 0 <= x <= 4.
#![allow(dead_code)]

use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, DummySolver, NlpProblem, Observer, Options, ProblemSpec,
    Result, SolveResult, SqpSolver,
};

/// Records every call so tests can check order and arguments.
#[derive(Default)]
pub struct Toy {
    pub calls: Vec<(&'static str, Vec<f64>)>,
}

impl NlpProblem for Toy {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> Result<f64> {
        self.calls.push(("functions", x.to_vec()));
        c[0] = x[0] + x[1];
        Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }

    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> Result<()> {
        self.calls.push(("derivatives", x.to_vec()));
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
        jac[0] = 1.0;
        jac[1] = 1.0;
        Ok(())
    }
}

pub fn spec() -> ProblemSpec {
    ProblemSpec::new(
        Bounds::new(vec![0.0, 0.0], vec![4.0, 4.0]),
        Bounds::fixed(vec![3.0]),
        CsrPattern::dense(1, 2),
    )
    .unwrap()
}

pub fn solve(
    problem: &mut dyn NlpProblem,
    x0: &[f64],
    cancel: &CancellationToken,
    observer: Option<&mut dyn Observer>,
) -> Result<SolveResult> {
    DummySolver.solve(&spec(), problem, x0, &Options::default(), cancel, observer)
}
