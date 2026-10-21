//! The stock benchmark models, registered through the public registry API
//! exactly as an application registers its own.
use crate::registry::{Instance, Model, ModelInfo, NoConfig, Registry};
use optimization_job_api::Result;
use serde_json::json;
use sqp_craft::{Bounds, CsrPattern, NlpProblem, ProblemSpec};

/// Sum of `(x[i]-(i+1)/n)^2`.
#[derive(Clone, Copy)]
struct Quadratic;
/// Sum of `x[i]^2`, subject to `sum(x) = 1`.
#[derive(Clone, Copy)]
struct Simplex;
/// The adjacent-pair Rosenbrock function.
#[derive(Clone, Copy)]
struct Rosenbrock;

impl NlpProblem for Quadratic {
    fn functions(&mut self, x: &[f64], _: &mut [f64]) -> sqp_craft::Result<f64> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, v)| (v - (i + 1) as f64 / x.len() as f64).powi(2))
            .sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> sqp_craft::Result<()> {
        for (i, v) in g.iter_mut().enumerate() {
            *v = 2.0 * (x[i] - (i + 1) as f64 / x.len() as f64);
        }
        Ok(())
    }
}
impl NlpProblem for Simplex {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> sqp_craft::Result<f64> {
        c[0] = x.iter().sum();
        Ok(x.iter().map(|x| x * x).sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> sqp_craft::Result<()> {
        for (i, v) in g.iter_mut().enumerate() {
            *v = 2.0 * x[i];
        }
        jac.fill(1.0);
        Ok(())
    }
}
impl NlpProblem for Rosenbrock {
    fn functions(&mut self, x: &[f64], _: &mut [f64]) -> sqp_craft::Result<f64> {
        Ok(x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> sqp_craft::Result<()> {
        g.fill(0.0);
        for i in 0..x.len() - 1 {
            let d = x[i + 1] - x[i] * x[i];
            g[i] += -400.0 * x[i] * d + 2.0 * (x[i] - 1.0);
            g[i + 1] += 200.0 * d;
        }
        Ok(())
    }
}

/// An unconstrained model on `[-5, 5]^n`.
fn boxed(n: usize) -> Result<ProblemSpec> {
    Ok(ProblemSpec::new(
        Bounds::new(vec![-5.0; n], vec![5.0; n]),
        Bounds::free(0),
        CsrPattern::dense(0, n),
    )?)
}

/// Registers the three benchmarks, version 1 each.
pub fn register(registry: &mut Registry) {
    registry
        .register(
            ModelInfo::new(
                "benchmark.quadratic",
                "1",
                "sum of (x[i]-(i+1)/n)^2, minimum 0",
            )
            .about("bounds", json!([-5, 5]))
            .example_dimension(4),
            |m: &Instance, _: NoConfig| Ok(Model::new(boxed(m.dimension())?, Quadratic)),
        )
        .register(
            ModelInfo::new(
                "benchmark.simplex",
                "1",
                "sum of x[i]^2 subject to sum(x)=1, minimum 1/n at x[i]=1/n",
            )
            .about("bounds", json!([0, 1]))
            .about("constraint", json!("sum(x)=1"))
            .example_dimension(4),
            |m: &Instance, _: NoConfig| {
                let n = m.dimension();
                let spec = ProblemSpec::new(
                    Bounds::new(vec![0.0; n], vec![1.0; n]),
                    Bounds::fixed(vec![1.0]),
                    CsrPattern::dense(1, n),
                )?;
                Ok(Model::new(spec, Simplex))
            },
        )
        .register(
            ModelInfo::new(
                "benchmark.rosenbrock",
                "1",
                "adjacent-pair Rosenbrock, minimum 0 at x[i]=1",
            )
            .dimensions(2..=256)
            .about("bounds", json!([-5, 5]))
            .example_dimension(2)
            .example_start(vec![-1.2, 1.0]),
            |m: &Instance, _: NoConfig| Ok(Model::new(boxed(m.dimension())?, Rosenbrock)),
        );
}
