//! `sqp.evaluate/1`: function values of given points, without solving.
use crate::{POINT_CHECKS, registry::Model};
use optimization_job_api::{
    Result,
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqp_craft::Options;

pub const PROFILE: &str = "sqp.evaluate/1";

/// `sqp.evaluate/1`: points to check against a registered model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub problem: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem_version: Option<String>,
    pub dimension: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    pub points: Vec<Vec<f64>>,
    pub feasibility_tolerance: f64,
}

pub(crate) struct Evaluate {
    model: Model,
    payload: Payload,
}

/// Checks a payload whose model is built; fills in the effective values.
pub(crate) fn prepare(
    mut p: Payload,
    built: crate::registry::Built,
) -> Result<(Payload, Evaluate)> {
    Options {
        threads: 1,
        feasibility_tolerance: p.feasibility_tolerance,
    }
    .validate()?;
    if p.points.is_empty() || p.points.len() > 1_000_000 {
        return Err("points must hold 1..1000000 points".into());
    }
    let n = built.model.spec().n();
    for (i, x) in p.points.iter().enumerate() {
        if x.len() != n || x.iter().any(|v| !v.is_finite()) {
            return Err(format!("point {i} needs {n} finite entries").into());
        }
    }
    p.problem_version = Some(built.version);
    p.config = built.config;
    Ok((
        p.clone(),
        Evaluate {
            model: built.model,
            payload: p,
        },
    ))
}

/// Objective, constraint values and the largest bound violation at `x`
/// from a fresh model instance, or why they are not finite.
fn values(model: &Model, x: &[f64]) -> Result<(f64, Vec<f64>, f64)> {
    let spec = model.spec();
    let mut fresh = model.instance();
    let mut rows = vec![0.0; spec.m()];
    let objective = fresh.functions(x, &mut rows)?;
    if !objective.is_finite() || rows.iter().any(|v| !v.is_finite()) {
        return Err("the objective or a constraint value is not finite".into());
    }
    let primal = spec
        .variables()
        .checked_violation(x)?
        .max(spec.constraints().checked_violation(&rows)?);
    Ok((objective, rows, primal))
}

impl Prepared for Evaluate {
    fn tasks(&self) -> usize {
        self.payload.points.len()
    }
    fn cancel(&self) {} // One evaluation per task; tasks stop between points.
    fn solve(&self, task: usize, _: f64) -> Result<TaskResult> {
        let x = &self.payload.points[task];
        let outcome = match values(&self.model, x) {
            Ok((objective, rows, primal)) => {
                let feasible = primal <= self.payload.feasibility_tolerance;
                Outcome {
                    claim: if feasible { "feasible" } else { "none" }.into(),
                    feasible: Some(feasible),
                    objective: Some(objective),
                    details: json!({"problem":self.payload.problem,"constraints":rows,
                        "primal_violation":primal,"feasibility_tolerance":self.payload.feasibility_tolerance}),
                    evidence: Some(Evidence::rechecked(&POINT_CHECKS)),
                    ..Outcome::default()
                }
            }
            // A finite point whose values overflow has no usable value; it
            // is an answer for this point, not a job failure.
            Err(error) => Outcome {
                claim: "none".into(),
                details: json!({"problem":self.payload.problem,"message":error.to_string()}),
                evidence: Some(Evidence::claimed(&[])),
                ..Outcome::default()
            },
        };
        Ok(TaskResult {
            outcome: Outcome {
                termination: "evaluated".into(),
                x: Some(x.clone()),
                ..outcome
            },
            work: [("evaluations".into(), 1)].into(),
        })
    }
}
