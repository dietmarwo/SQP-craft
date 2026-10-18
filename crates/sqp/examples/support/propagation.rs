//! Independent controlled nonlinear ODE segments with forward sensitivities.
use sqp_craft::{Error, NlpProblem, Result};
use sqp_craft_parallel::Execution;
use std::thread::{self, ThreadId};

pub struct Propagation {
    execution: Execution,
    owner: ThreadId,
    forcing: Vec<[f64; 3]>,
    target: Vec<f64>,
    propagated: Vec<(f64, f64)>,
}

pub fn control(i: usize) -> f64 {
    0.7 + (i % 5) as f64 * 0.05
}
fn rate(i: usize) -> f64 {
    0.5 + (i % 7) as f64 * 0.1
}
fn forcing(steps: usize) -> Vec<[f64; 3]> {
    let dt = 2.0 / steps as f64;
    (0..steps)
        .map(|k| {
            let t = k as f64 * dt;
            [t, t + 0.5 * dt, t + dt].map(|t| 1.0 + 0.05 * t.sin())
        })
        .collect()
}
fn rhs(y: f64, z: f64, u: f64, rate: f64, force: f64) -> (f64, f64) {
    (
        -rate * y - 0.03 * y * y * y + u * force,
        (-rate - 0.09 * y * y) * z + force,
    )
}
fn integrate(u: f64, rate: f64, forcing: &[[f64; 3]]) -> (f64, f64) {
    let dt = 2.0 / forcing.len() as f64;
    let (mut y, mut z) = (0.0, 0.0);
    for &[a, b, c] in forcing {
        let k1 = rhs(y, z, u, rate, a);
        let k2 = rhs(y + 0.5 * dt * k1.0, z + 0.5 * dt * k1.1, u, rate, b);
        let k3 = rhs(y + 0.5 * dt * k2.0, z + 0.5 * dt * k2.1, u, rate, b);
        let k4 = rhs(y + dt * k3.0, z + dt * k3.1, u, rate, c);
        y += dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        z += dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
    }
    (y, z)
}
impl Propagation {
    pub fn new(n: usize, steps: usize) -> Self {
        let fine = forcing(steps * 2);
        let target = (0..n)
            .map(|i| integrate(control(i), rate(i), &fine).0)
            .collect();
        Self {
            execution: Execution::default(),
            owner: thread::current().id(),
            forcing: forcing(steps),
            target,
            propagated: vec![(0.0, 0.0); n],
        }
    }
}
impl NlpProblem for Propagation {
    fn set_threads(&mut self, threads: usize) -> Result<()> {
        assert_eq!(thread::current().id(), self.owner);
        self.execution = Execution::new(threads).map_err(Error::Unsupported)?;
        Ok(())
    }
    fn functions(&mut self, x: &[f64], constraints: &mut [f64]) -> Result<f64> {
        assert_eq!(thread::current().id(), self.owner);
        assert!(constraints.is_empty());
        let forcing = &self.forcing;
        self.execution.chunks_mut(
            &mut self.propagated,
            1,
            x.len() * forcing.len() * 64,
            |i, out| out[0] = integrate(x[i], rate(i), forcing),
        );
        Ok(self
            .propagated
            .iter()
            .zip(&self.target)
            .map(|(&(y, _), &target)| 0.5 * (y - target).powi(2))
            .sum())
    }
    fn derivatives(&mut self, _: &[f64], gradient: &mut [f64], jacobian: &mut [f64]) -> Result<()> {
        assert_eq!(thread::current().id(), self.owner);
        assert!(jacobian.is_empty());
        for ((g, &(y, z)), &target) in gradient.iter_mut().zip(&self.propagated).zip(&self.target) {
            *g = (y - target) * z;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rk4_sensitivities_match_finite_differences_and_refinement() {
        let f = forcing(1024);
        let fine = forcing(2048);
        for u in [0.05, 0.7, 2.0] {
            for rate in [0.5, 1.1] {
                let (y, z) = integrate(u, rate, &f);
                let numerical =
                    (integrate(u + 1e-6, rate, &f).0 - integrate(u - 1e-6, rate, &f).0) / 2e-6;
                assert!((z - numerical).abs() < 1e-8);
                assert!((y - integrate(u, rate, &fine).0).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn serial_and_eight_segment_outputs_are_bit_identical() {
        let n = 32;
        let mut a = Propagation::new(n, 1024);
        let mut b = Propagation::new(n, 1024);
        b.set_threads(8).unwrap();
        let x = vec![0.4; n];
        assert_eq!(
            a.functions(&x, &mut []).unwrap(),
            b.functions(&x, &mut []).unwrap()
        );
        assert_eq!(a.propagated, b.propagated);
        let mut ga = vec![0.; n];
        let mut gb = ga.clone();
        a.derivatives(&x, &mut ga, &mut []).unwrap();
        b.derivatives(&x, &mut gb, &mut []).unwrap();
        assert_eq!(ga, gb);
    }
}
