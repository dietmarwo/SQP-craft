//! Native registered smooth models; callbacks never cross the job boundary.
#![forbid(unsafe_code)]
use optimization_job_api::{
    JobSpec, Result,
    result::{Evidence, Outcome},
    worker::{self, Prepared, TaskResult, Worker},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqp_craft::{
    Bounds, CancellationToken, CsrPattern, NlpProblem, Options, ProblemSpec, SolveResult,
    SqpSolver, Status,
    kkt::{self, KktPoint},
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};
use std::path::Path;

/// The engine version: this crate shares the workspace version with
/// `sqp-craft-solver`, so the reported version cannot drift from the linked
/// solver.
const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROFILE: &str = "sqp.solve/1";
const EVALUATE_PROFILE: &str = "sqp.evaluate/1";
/// KKT acceptance of `first_order`, besides the payload's tolerances.
const COMPLEMENTARITY: f64 = 1e-6;
const DUAL_SIGN: f64 = 1e-8;
/// Properties recomputed when a returned point is re-evaluated.
const POINT_CHECKS: [&str; 3] = ["objective", "bounds", "constraints"];

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
/// `sqp.evaluate/1`: points to check against a registered model.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatePayload {
    problem: String,
    dimension: usize,
    points: Vec<Vec<f64>>,
    feasibility_tolerance: f64,
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
struct Evaluate {
    model: Model,
    spec: ProblemSpec,
    payload: EvaluatePayload,
}
/// A point recomputed from the registered model, outside the solver.
struct Recomputed {
    objective: f64,
    rows: Vec<f64>,
    gradient: Vec<f64>,
    jacobian: Vec<f64>,
    /// Largest violation of the variable and constraint bounds.
    primal: f64,
}
/// Objective, constraint values and the largest bound violation at `x`, or
/// why they are not finite.
fn values(model: &Model, spec: &ProblemSpec, x: &[f64]) -> Result<(f64, Vec<f64>, f64)> {
    let mut fresh = model.clone();
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
fn recompute(model: &Model, spec: &ProblemSpec, x: &[f64]) -> Result<Recomputed> {
    let mut fresh = model.clone();
    let mut rows = vec![0.0; spec.m()];
    let objective = fresh.functions(x, &mut rows)?;
    let mut gradient = vec![0.0; spec.n()];
    let mut jacobian = vec![0.0; spec.nnz()];
    fresh.derivatives(x, &mut gradient, &mut jacobian)?;
    if !objective.is_finite()
        || rows
            .iter()
            .chain(&gradient)
            .chain(&jacobian)
            .any(|v| !v.is_finite())
    {
        return Err("nonfinite independent model evaluation".into());
    }
    let primal = spec
        .variables()
        .checked_violation(x)?
        .max(spec.constraints().checked_violation(&rows)?);
    Ok(Recomputed {
        objective,
        rows,
        gradient,
        jacobian,
        primal,
    })
}
impl Worker for Adapter {
    fn describe(&self) -> Value {
        let mut d = worker::description(
            "sqp-craft",
            ENGINE_VERSION,
            json!([PROFILE, EVALUATE_PROFILE]),
            json!([{"id":"benchmark.quadratic","dimension":"1..256","bounds":[-5,5]},{"id":"benchmark.simplex","dimension":"1..256","bounds":[0,1],"constraint":"sum(x)=1"},{"id":"benchmark.rosenbrock","dimension":"2..256","bounds":[-5,5]}]),
            8,
        );
        d["backends"] = json!(["active_set", "clarabel"]);
        d["cancellation"] = json!("native cooperative token plus runner process deadline");
        d["claims"] = json!({
            "first_order":"converged, and the KKT residuals recomputed from the registered model pass the acceptance; no global or second-order certificate",
            "feasible":"a re-evaluated point within the feasibility tolerance, without an optimality claim (also what survives a refuted first_order claim)",
            "locally_infeasible":"elastic mode stopped at a stationary point of the l1 violation; the point's infeasibility is rechecked, its stationarity is the solver's",
            "infeasible":"linear rows and variable bounds have no common point, as judged by the QP solver (claimed)",
            "none":"no claim; evidence.refuted names a converged status that failed its re-check"
        });
        d["kkt_acceptance"] = json!({"primal":"feasibility_tolerance","stationarity":"stationarity_tolerance",
            "complementarity":COMPLEMENTARITY,"dual_sign":DUAL_SIGN});
        d["evaluation"] = json!({"profile":EVALUATE_PROFILE,"inner_threads":1,
            "points":"1 or more finite points of the problem's dimension, also outside the bounds (inline: the resolved job, points included, must stay below the runner's 4 MiB validation output); one task per point; a point whose values overflow is reported with claim none",
            "result":"claim feasible within feasibility_tolerance, else none; objective, constraint values and violation"});
        d
    }
    fn prepare(&self, s: &mut JobSpec, _: &Path) -> Result<Box<dyn Prepared>> {
        if s.payload_schema == EVALUATE_PROFILE {
            return prepare_evaluate(s);
        }
        if s.payload_schema != PROFILE || s.resources.inner_threads > 8 || !s.inputs.is_empty() {
            return Err(format!(
                "{PROFILE} supports inner_threads=1..8 and registered models without inputs; {EVALUATE_PROFILE} evaluates points"
            )
            .into());
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
fn prepare_evaluate(s: &mut JobSpec) -> Result<Box<dyn Prepared>> {
    if s.resources.inner_threads != 1 || !s.inputs.is_empty() {
        return Err(format!(
            "{EVALUATE_PROFILE} evaluates serially without inputs; use inner_threads=1"
        )
        .into());
    }
    let p: EvaluatePayload = serde_json::from_value(s.payload.clone())?;
    let (model, spec) = registered(&p.problem, p.dimension)?;
    Options {
        threads: 1,
        feasibility_tolerance: p.feasibility_tolerance,
    }
    .validate()?;
    if p.points.is_empty() || p.points.len() > 1_000_000 {
        return Err("points must hold 1..1000000 points".into());
    }
    for (i, x) in p.points.iter().enumerate() {
        if x.len() != spec.n() || x.iter().any(|v| !v.is_finite()) {
            return Err(format!("point {i} needs {} finite entries", spec.n()).into());
        }
    }
    s.payload = serde_json::to_value(&p)?;
    Ok(Box::new(Evaluate {
        model,
        spec,
        payload: p,
    }))
}
impl Run {
    /// Judges a native result. A claim whose independent check fails is
    /// refuted, never reported: converged without passing the recomputed
    /// KKT acceptance (`false_optimality`, or `infeasible_point` for an
    /// infeasible point), or an infeasibility claim at a feasible point
    /// (`false_infeasibility`).
    fn judge(&self, r: SolveResult) -> Result<TaskResult> {
        let tolerance = self.options.feasibility_tolerance;
        let mut recomputed = None;
        let mut residuals = Value::Null;
        let mut kkt_checked = false;
        let mut accepted = false;
        let mut why = String::new();
        if let Some(x) = &r.candidate {
            self.spec.check_point(x, "returned candidate")?;
            let p = recompute(&self.model, &self.spec, x)?;
            if let Some(multipliers) = &r.multipliers {
                let k = kkt::residuals(
                    &KktPoint {
                        x,
                        variable_bounds: self.spec.variables(),
                        rows: &p.rows,
                        row_bounds: self.spec.constraints(),
                        gradient: &p.gradient,
                        jacobian: self.spec.jacobian(),
                        jacobian_values: &p.jacobian,
                    },
                    multipliers,
                )?;
                kkt_checked = true;
                accepted = p.primal <= tolerance
                    && k.stationarity <= self.payload.stationarity_tolerance
                    && k.complementarity <= COMPLEMENTARITY
                    && k.dual_sign <= DUAL_SIGN;
                why = format!(
                    "primal {} (tolerance {tolerance}), stationarity {} ({}), complementarity {} ({COMPLEMENTARITY}), dual_sign {} ({DUAL_SIGN})",
                    p.primal,
                    k.stationarity,
                    self.payload.stationarity_tolerance,
                    k.complementarity,
                    k.dual_sign
                );
                residuals = json!({"primal":k.primal,"stationarity":k.stationarity,"complementarity":k.complementarity,"dual_sign":k.dual_sign});
            }
            recomputed = Some(p);
        }
        let feasible = recomputed.as_ref().map(|p| p.primal <= tolerance);
        let mut checks = POINT_CHECKS.to_vec();
        if kkt_checked {
            checks.push("kkt");
        }
        let point: &[&str] = if recomputed.is_some() {
            &POINT_CHECKS
        } else {
            &[]
        };
        let mut note = None;
        let (claim, evidence) = match r.status {
            Status::Converged if accepted => ("first_order", Evidence::rechecked(&checks)),
            Status::Converged if feasible == Some(false) => (
                "none",
                Evidence::rechecked(&checks).refuting("first_order", "infeasible_point", why),
            ),
            Status::Converged if kkt_checked => (
                "feasible",
                Evidence::rechecked(&checks).refuting("first_order", "false_optimality", why),
            ),
            // Converged, but nothing to check the first-order conditions
            // with: not confirmed, and not refuted either.
            Status::Converged if feasible == Some(true) => {
                note = Some("converged without multipliers: first-order conditions not checked");
                ("feasible", Evidence::rechecked(&POINT_CHECKS))
            }
            Status::LocallyInfeasible | Status::LinearInfeasible if feasible == Some(true) => {
                let refuted = if r.status == Status::LocallyInfeasible {
                    "locally_infeasible"
                } else {
                    "infeasible"
                };
                (
                    "feasible",
                    Evidence::rechecked(&POINT_CHECKS).refuting(
                        refuted,
                        "false_infeasibility",
                        "the returned point satisfies every bound and constraint within the tolerance",
                    ),
                )
            }
            // The point was recomputed and is infeasible, but the claim, a
            // stationary point of the violation, is the solver's.
            Status::LocallyInfeasible => ("locally_infeasible", Evidence::claimed(point)),
            Status::LinearInfeasible => ("infeasible", Evidence::claimed(point)),
            _ if feasible == Some(true) => ("feasible", Evidence::rechecked(&POINT_CHECKS)),
            _ => ("none", Evidence::claimed(point)),
        };
        let validation_calls = if recomputed.is_some() { 2 } else { 0 };
        let mut details = json!({"kkt":residuals,"primal_violation":recomputed.as_ref().map(|p| p.primal),"qp_backend":r.report.qp_backend,"qp_status":r.report.qp_status,"message":r.message,"global_optimum_certified":false,"acceptance":{"primal":tolerance,"stationarity":self.payload.stationarity_tolerance,"complementarity":COMPLEMENTARITY,"dual_sign":DUAL_SIGN}});
        if let Some(note) = note {
            details["note"] = json!(note);
        }
        Ok(TaskResult {
            outcome: Outcome {
                termination: r.status.as_str().into(),
                claim: claim.into(),
                feasible,
                objective: recomputed.as_ref().map(|p| p.objective),
                x: r.candidate,
                details,
                evidence: Some(evidence),
                ..Outcome::default()
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
        self.judge(r)
    }
}
impl Prepared for Evaluate {
    fn tasks(&self) -> usize {
        self.payload.points.len()
    }
    fn cancel(&self) {} // One evaluation per task; tasks stop between points.
    fn solve(&self, task: usize, _: f64) -> Result<TaskResult> {
        let x = &self.payload.points[task];
        let outcome = match values(&self.model, &self.spec, x) {
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
fn main() {
    worker::run(Adapter);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqp_craft::Multipliers;

    /// A worker run on a registered model, for judging synthetic results.
    fn run(problem: &str, dimension: usize) -> Run {
        let (model, spec) = registered(problem, dimension).unwrap();
        Run {
            model,
            spec,
            payload: Payload {
                problem: problem.into(),
                dimension,
                starts: vec![vec![0.5; dimension]],
                backend: "active_set".into(),
                max_iterations: 100,
                max_callback_calls: 1000,
                feasibility_tolerance: 1e-7,
                stationarity_tolerance: 1e-6,
            },
            options: Options {
                threads: 1,
                feasibility_tolerance: 1e-7,
            },
            cancel: CancellationToken::default(),
        }
    }
    fn native(
        status: Status,
        x: Option<Vec<f64>>,
        multipliers: Option<Multipliers>,
    ) -> SolveResult {
        SolveResult {
            report: Default::default(),
            dual_sign: None,
            status,
            candidate: x,
            evaluation: None,
            multipliers,
            counts: Default::default(),
            iterations: 0,
            stationarity: None,
            complementarity: None,
            message: String::new(),
        }
    }
    fn judged(r: &Run, result: SolveResult) -> Outcome {
        let o = r.judge(result).unwrap().outcome;
        o.check().unwrap();
        o
    }
    fn refuted(o: &Outcome) -> (&str, &str, &str) {
        let r = o.evidence.as_ref().unwrap().refuted.as_ref().unwrap();
        (o.claim.as_str(), r.claim.as_str(), r.verdict.as_str())
    }

    #[test]
    fn converged_and_infeasibility_claims_are_rechecked_or_refuted() {
        // min x0^2 + x1^2 subject to x0 + x1 = 1: optimum (0.5, 0.5) with
        // constraint multiplier 1 and inactive bounds.
        let simplex = run("benchmark.simplex", 2);
        let right = || Multipliers {
            constraints: vec![1.0],
            variables: vec![0.0, 0.0],
        };
        let o = judged(
            &simplex,
            native(Status::Converged, Some(vec![0.5, 0.5]), Some(right())),
        );
        assert_eq!(o.claim, "first_order");
        let e = o.evidence.as_ref().unwrap();
        assert_eq!((e.level.as_str(), e.checks.len()), ("rechecked", 4));
        assert_eq!(o.objective, Some(0.5));
        let zero = Multipliers {
            constraints: vec![0.0],
            variables: vec![0.0, 0.0],
        };
        let o = judged(
            &simplex,
            native(Status::Converged, Some(vec![0.5, 0.5]), Some(zero)),
        );
        assert_eq!(refuted(&o), ("feasible", "first_order", "false_optimality"));
        assert!(
            o.evidence
                .as_ref()
                .unwrap()
                .refuted
                .as_ref()
                .unwrap()
                .message
                .contains("stationarity 1")
        );
        let o = judged(
            &simplex,
            native(Status::Converged, Some(vec![0.2, 0.2]), Some(right())),
        );
        assert_eq!(refuted(&o), ("none", "first_order", "infeasible_point"));
        assert_eq!(o.feasible, Some(false));
        // Elastic mode at a feasible point: the infeasibility claim is false.
        let o = judged(
            &simplex,
            native(Status::LocallyInfeasible, Some(vec![0.5, 0.5]), None),
        );
        assert_eq!(
            refuted(&o),
            ("feasible", "locally_infeasible", "false_infeasibility")
        );
        let o = judged(
            &simplex,
            native(Status::LocallyInfeasible, Some(vec![0.2, 0.2]), None),
        );
        assert_eq!(
            (o.claim.as_str(), o.feasible),
            ("locally_infeasible", Some(false))
        );
        // The point's infeasibility is recomputed; the claim stays the solver's.
        let e = o.evidence.as_ref().unwrap();
        assert_eq!((e.level.as_str(), e.checks.len()), ("claimed", 3));
        let o = judged(&simplex, native(Status::LinearInfeasible, None, None));
        assert_eq!(o.claim, "infeasible");
        assert_eq!(o.evidence.as_ref().unwrap().level, "claimed");
        let o = judged(
            &simplex,
            native(Status::LinearInfeasible, Some(vec![0.5, 0.5]), None),
        );
        assert_eq!(
            refuted(&o),
            ("feasible", "infeasible", "false_infeasibility")
        );
        // Converged without multipliers: neither confirmed nor refuted.
        let o = judged(
            &simplex,
            native(Status::Converged, Some(vec![0.5, 0.5]), None),
        );
        assert_eq!(o.claim, "feasible");
        assert!(o.evidence.as_ref().unwrap().refuted.is_none());
        assert!(o.details["note"].is_string());
        let o = judged(
            &simplex,
            native(Status::Converged, Some(vec![0.2, 0.2]), None),
        );
        assert_eq!(refuted(&o), ("none", "first_order", "infeasible_point"));
        let o = judged(
            &simplex,
            native(Status::IterationLimit, Some(vec![0.6, 0.4]), None),
        );
        assert_eq!(
            (o.termination.as_str(), o.claim.as_str()),
            ("iteration_limit", "feasible")
        );
    }

    #[test]
    fn evaluate_checks_points_inside_and_outside_the_bounds() {
        let mut s: JobSpec = serde_json::from_value(json!({"schema_version":"optimization.job/0.1",
            "request_key":"test","worker":"sqp","payload_schema":EVALUATE_PROFILE,
            "payload":{"problem":"benchmark.simplex","dimension":2,"points":[[0.25,0.75],[0.5,0.25],[2.0,-1.0],[1e200,0.0]],
                "feasibility_tolerance":1e-7},
            "resources":{"cpu_slots":1,"outer_workers":1,"inner_threads":1},
            "limits":{"wall_time_seconds":10.0,"grace_seconds":1.0},"inputs":[]}))
        .unwrap();
        let p = Adapter.prepare(&mut s, Path::new(".")).unwrap();
        let outcomes: Vec<Outcome> = (0..4).map(|t| p.solve(t, 1.0).unwrap().outcome).collect();
        assert_eq!(
            outcomes
                .iter()
                .map(|o| o.claim.as_str())
                .collect::<Vec<_>>(),
            ["feasible", "none", "none", "none"]
        );
        // The objective overflows: no value, an answer for this point only.
        assert!(outcomes[3].objective.is_none() && outcomes[3].feasible.is_none());
        assert_eq!(outcomes[3].evidence.as_ref().unwrap().level, "claimed");
        assert_eq!(outcomes[0].objective, Some(0.625));
        assert_eq!(outcomes[1].details["primal_violation"], 0.25);
        // Outside the variable bounds: evaluated, and infeasible by 1.
        assert_eq!(outcomes[2].details["primal_violation"], 1.0);
        for o in &outcomes {
            assert_eq!(o.termination, "evaluated");
            o.check().unwrap();
        }
        for (key, value, message) in [
            ("points", json!([]), "1..1000000"),
            ("points", json!([[1.0]]), "point 0"),
            ("feasibility_tolerance", json!(0.0), "tolerance"),
        ] {
            let mut bad = s.clone();
            bad.payload[key] = value;
            let err = Adapter
                .prepare(&mut bad, Path::new("."))
                .err()
                .unwrap()
                .to_string();
            assert!(err.contains(message), "{message}: {err}");
        }
    }

    #[test]
    fn description_reports_the_workspace_engine_version_and_profiles() {
        let d = Adapter.describe();
        assert_eq!(d["protocol"], "optimization.worker/0.2");
        assert_eq!(d["profiles"], json!([PROFILE, EVALUATE_PROFILE]));
        let manifest = include_str!("../../../Cargo.toml");
        let workspace = &manifest[manifest.find("[workspace.package]").unwrap()..];
        let version = workspace
            .lines()
            .find_map(|l| l.strip_prefix("version = "))
            .unwrap()
            .trim_matches('"');
        assert_eq!(d["build"]["engine_version"], version);
    }
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
