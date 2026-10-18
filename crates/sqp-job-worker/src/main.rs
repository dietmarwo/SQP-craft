//! Native registered smooth models; callbacks never cross the job boundary.
#![forbid(unsafe_code)]
use optimization_job_api::{
    JobSpec, Result,
    result::Outcome,
    worker::{self, Prepared, TaskResult, Worker},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, SqpSolver, Status,
    kkt::{self, KktPoint},
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};
use std::path::Path;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    problem: String,
    dimension: usize,
    starts: Vec<Vec<f64>>,
    backend: String,
    max_iterations: u64,
    max_callback_calls: u64,
    feasibility_tolerance: f64,
    stationarity_tolerance: f64,
}
#[derive(Clone)]
enum Model {
    Quadratic,
    Simplex,
    Rosenbrock,
}
impl NlpProblem for Model {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> sqp_craft::Result<f64> {
        Ok(match self {
            Self::Quadratic => x
                .iter()
                .enumerate()
                .map(|(i, v)| (v - (i + 1) as f64 / x.len() as f64).powi(2))
                .sum(),
            Self::Simplex => {
                c[0] = x.iter().sum();
                x.iter().map(|x| x * x).sum()
            }
            Self::Rosenbrock => x
                .windows(2)
                .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
                .sum(),
        })
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> sqp_craft::Result<()> {
        match self {
            Self::Quadratic => {
                for (i, v) in g.iter_mut().enumerate() {
                    *v = 2.0 * (x[i] - (i + 1) as f64 / x.len() as f64);
                }
            }
            Self::Simplex => {
                for (i, v) in g.iter_mut().enumerate() {
                    *v = 2.0 * x[i];
                }
                jac.fill(1.0);
            }
            Self::Rosenbrock => {
                g.fill(0.0);
                for i in 0..x.len() - 1 {
                    let d = x[i + 1] - x[i] * x[i];
                    g[i] += -400.0 * x[i] * d + 2.0 * (x[i] - 1.0);
                    g[i + 1] += 200.0 * d;
                }
            }
        };
        Ok(())
    }
}
fn registered(id: &str, n: usize) -> Result<(Model, ProblemSpec)> {
    if !(1..=256).contains(&n) {
        return Err("native SQP dimension must lie in 1..=256".into());
    }
    Ok(match id {
        "benchmark.quadratic" => (
            Model::Quadratic,
            ProblemSpec::new(
                Bounds::new(vec![-5.0; n], vec![5.0; n]),
                Bounds::free(0),
                CsrPattern::dense(0, n),
            )?,
        ),
        "benchmark.simplex" => (
            Model::Simplex,
            ProblemSpec::new(
                Bounds::new(vec![0.0; n], vec![1.0; n]),
                Bounds::fixed(vec![1.0]),
                CsrPattern::dense(1, n),
            )?,
        ),
        "benchmark.rosenbrock" if n >= 2 => (
            Model::Rosenbrock,
            ProblemSpec::new(
                Bounds::new(vec![-5.0; n], vec![5.0; n]),
                Bounds::free(0),
                CsrPattern::dense(0, n),
            )?,
        ),
        _ => return Err("unknown registered problem or invalid Rosenbrock dimension".into()),
    })
}
struct Adapter;
struct Run {
    model: Model,
    spec: ProblemSpec,
    payload: Payload,
    options: Options,
    cancel: CancellationToken,
}
impl Worker for Adapter {
    fn describe(&self) -> Value {
        let mut d = worker::description(
            "sqp-craft",
            "0.16.0",
            json!(["sqp.solve/1"]),
            json!([{"id":"benchmark.quadratic","dimension":"1..256","bounds":[-5,5]},{"id":"benchmark.simplex","dimension":"1..256","bounds":[0,1],"constraint":"sum(x)=1"},{"id":"benchmark.rosenbrock","dimension":"2..256","bounds":[-5,5]}]),
            8,
        );
        d["backends"] = json!(["active_set", "clarabel"]);
        d["cancellation"] = json!("native cooperative token plus runner process deadline");
        d
    }
    fn prepare(&self, s: &mut JobSpec, _: &Path) -> Result<Box<dyn Prepared>> {
        if s.payload_schema != "sqp.solve/1"
            || s.resources.inner_threads > 8
            || !s.inputs.is_empty()
        {
            return Err(
                "sqp.solve/1 supports inner_threads=1..8 and registered models without inputs"
                    .into(),
            );
        }
        let p: Payload = serde_json::from_value(s.payload.clone())?;
        if !matches!(p.backend.as_str(), "active_set" | "clarabel")
            || p.starts.is_empty()
            || p.starts.len() > 1_000_000
            || p.max_callback_calls == 0
        {
            return Err("invalid backend, starts or callback budget".into());
        }
        let (model, spec) = registered(&p.problem, p.dimension)?;
        for start in &p.starts {
            spec.check_point(start, "initial point")?;
        }
        let options = Options {
            threads: s.resources.inner_threads,
            feasibility_tolerance: p.feasibility_tolerance,
        };
        options.validate()?;
        SqpOptions {
            stationarity_tolerance: p.stationarity_tolerance,
            ..Default::default()
        }
        .validate()?;
        s.payload = serde_json::to_value(&p)?;
        Ok(Box::new(Run {
            model,
            spec,
            payload: p,
            options,
            cancel: CancellationToken::default(),
        }))
    }
}
impl Prepared for Run {
    fn tasks(&self) -> usize {
        self.payload.starts.len()
    }
    fn cancel(&self) {
        self.cancel.cancel();
    }
    fn solve(&self, task: usize, remaining: f64) -> Result<TaskResult> {
        let mut model = self.model.clone();
        let settings = SqpOptions {
            max_iterations: self.payload.max_iterations,
            max_callback_calls: self.payload.max_callback_calls,
            time_limit_seconds: Some(remaining),
            stationarity_tolerance: self.payload.stationarity_tolerance,
            ..SqpOptions::default()
        };
        let r = if self.payload.backend == "active_set" {
            NativeSqp {
                qp: ActiveSetQp::default(),
                settings,
            }
            .solve(
                &self.spec,
                &mut model,
                &self.payload.starts[task],
                &self.options,
                &self.cancel,
                None,
            )?
        } else {
            NativeSqp {
                qp: ClarabelQp::default(),
                settings,
            }
            .solve(
                &self.spec,
                &mut model,
                &self.payload.starts[task],
                &self.options,
                &self.cancel,
                None,
            )?
        };
        let mut objective = None;
        let mut feasible = None;
        let mut residuals = Value::Null;
        let mut accepted = false;
        let mut validation_calls = 0;
        if let Some(x) = &r.candidate {
            self.spec.check_point(x, "returned candidate")?;
            let mut fresh = self.model.clone();
            let mut rows = vec![0.0; self.spec.m()];
            let f = fresh.functions(x, &mut rows)?;
            let mut gradient = vec![0.0; self.spec.n()];
            let mut jac = vec![0.0; self.spec.nnz()];
            fresh.derivatives(x, &mut gradient, &mut jac)?;
            validation_calls = 2;
            if !f.is_finite()
                || rows
                    .iter()
                    .chain(&gradient)
                    .chain(&jac)
                    .any(|v| !v.is_finite())
            {
                return Err("nonfinite independent model evaluation".into());
            }
            objective = Some(f);
            let primal = self
                .spec
                .variables()
                .checked_violation(x)?
                .max(self.spec.constraints().checked_violation(&rows)?);
            feasible = Some(primal <= self.options.feasibility_tolerance);
            if let Some(multipliers) = &r.multipliers {
                let k = kkt::residuals(
                    &KktPoint {
                        x,
                        variable_bounds: self.spec.variables(),
                        rows: &rows,
                        row_bounds: self.spec.constraints(),
                        gradient: &gradient,
                        jacobian: self.spec.jacobian(),
                        jacobian_values: &jac,
                    },
                    multipliers,
                )?;
                accepted = feasible == Some(true)
                    && k.stationarity <= self.payload.stationarity_tolerance
                    && k.complementarity <= 1e-6
                    && k.dual_sign <= 1e-8;
                residuals = json!({"primal":k.primal,"stationarity":k.stationarity,"complementarity":k.complementarity,"dual_sign":k.dual_sign});
            }
        }
        if r.status == Status::Converged && !accepted {
            return Err("converged result failed independent KKT acceptance".into());
        }
        let claim = if r.status == Status::Converged {
            "first_order"
        } else if feasible == Some(true) {
            "feasible"
        } else {
            "none"
        };
        Ok(TaskResult {
            outcome: Outcome {
                termination: r.status.as_str().into(),
                claim: claim.into(),
                feasible,
                objective,
                x: r.candidate,
                details: json!({"kkt":residuals,"qp_backend":r.report.qp_backend,"qp_status":r.report.qp_status,"message":r.message,"global_optimum_certified":false,"acceptance":{"primal":self.options.feasibility_tolerance,"stationarity":self.payload.stationarity_tolerance,"complementarity":1e-6,"dual_sign":1e-8}}),
            },
            work: [
                ("functions".into(), r.counts.functions),
                ("derivatives".into(), r.counts.derivatives),
                ("iterations".into(), r.iterations),
                ("qp_iterations".into(), r.report.qp_iterations),
                ("validation_calls".into(), validation_calls),
            ]
            .into(),
        })
    }
}
fn main() {
    worker::run(Adapter);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registered_derivatives_match_finite_differences() {
        for id in [
            "benchmark.quadratic",
            "benchmark.simplex",
            "benchmark.rosenbrock",
        ] {
            let (mut m, spec) = registered(id, 4).unwrap();
            let report = sqp_craft::derivative_check::check_derivatives(
                &spec,
                &mut m,
                &[0.2, 0.3, 0.1, 0.4],
                &Default::default(),
            )
            .unwrap();
            assert!(report.passed, "{id}: {report:?}");
        }
    }
}
