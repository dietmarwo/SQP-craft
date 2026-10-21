//! `sqp.solve/1`: one native SQP solve per start, with an independent KKT
//! recheck of every result.
use crate::{COMPLEMENTARITY, DUAL_SIGN, POINT_CHECKS, registry::Model};
use optimization_job_api::{
    Result,
    result::{Evidence, Outcome},
    worker::{Prepared, TaskResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqp_craft::{
    CancellationToken, Options, ProblemSpec, SolveResult, SqpSolver, Status,
    kkt::{self, KktPoint},
};
use sqp_craft_activeset::ActiveSetQp;
use sqp_craft_clarabel::ClarabelQp;
use sqp_craft_solver::{NativeSqp, SqpOptions};

pub const PROFILE: &str = "sqp.solve/1";
/// The QP backends a job can choose.
pub const BACKENDS: [&str; 2] = ["active_set", "clarabel"];

/// The `sqp.solve/1` payload. The optional fields default to the
/// registered model version, the model's default config and elastic mode
/// off; the resolved job records all three.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub problem: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem_version: Option<String>,
    pub dimension: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    pub starts: Vec<Vec<f64>>,
    pub backend: String,
    #[serde(default)]
    pub elastic_mode: bool,
    pub max_iterations: u64,
    pub max_callback_calls: u64,
    pub feasibility_tolerance: f64,
    pub stationarity_tolerance: f64,
}

pub(crate) struct Run {
    pub model: Model,
    pub payload: Payload,
    pub options: Options,
    pub cancel: CancellationToken,
}

/// Checks a payload whose model is built; fills in the effective values.
pub(crate) fn prepare(
    mut p: Payload,
    built: crate::registry::Built,
    inner_threads: usize,
) -> Result<(Payload, Run)> {
    if !BACKENDS.contains(&p.backend.as_str())
        || p.starts.is_empty()
        || p.starts.len() > 1_000_000
        || p.max_callback_calls == 0
    {
        return Err("invalid backend, starts or callback budget".into());
    }
    if p.elastic_mode && p.backend != "active_set" {
        return Err(
            "elastic_mode needs backend active_set: the Clarabel backend has no elastic QP".into(),
        );
    }
    for start in &p.starts {
        built.model.spec().check_point(start, "initial point")?;
    }
    let options = Options {
        threads: inner_threads,
        feasibility_tolerance: p.feasibility_tolerance,
    };
    options.validate()?;
    SqpOptions {
        stationarity_tolerance: p.stationarity_tolerance,
        elastic_mode: p.elastic_mode,
        ..Default::default()
    }
    .validate()?;
    p.problem_version = Some(built.version);
    p.config = built.config;
    Ok((
        p.clone(),
        Run {
            model: built.model,
            payload: p,
            options,
            cancel: CancellationToken::default(),
        },
    ))
}

/// A point recomputed from a fresh model instance, outside the solver.
struct Recomputed {
    objective: f64,
    rows: Vec<f64>,
    gradient: Vec<f64>,
    jacobian: Vec<f64>,
    /// Largest violation of the variable and constraint bounds.
    primal: f64,
}

/// Recomputes the model at `x` (inside the variable bounds) with a fresh
/// instance; `calls` counts the model calls made, also when they fail.
fn recompute(model: &Model, x: &[f64], calls: &mut u64) -> Result<Recomputed> {
    let spec = model.spec();
    let mut fresh = model.instance();
    let mut rows = vec![0.0; spec.m()];
    *calls += 1;
    let objective = fresh.functions(x, &mut rows)?;
    let mut gradient = vec![0.0; spec.n()];
    let mut jacobian = vec![0.0; spec.nnz()];
    *calls += 1;
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

impl Run {
    fn spec(&self) -> &ProblemSpec {
        self.model.spec()
    }
    /// The options the task ran with, as the result reports them.
    fn effective_options(&self) -> Value {
        json!({"backend":self.payload.backend,"elastic_mode":self.payload.elastic_mode,
            "max_iterations":self.payload.max_iterations,"max_callback_calls":self.payload.max_callback_calls,
            "threads":self.options.threads})
    }
    /// Judges a native result. A claim whose independent check fails is
    /// refuted, never reported: converged without passing the recomputed
    /// KKT acceptance, also when the residuals overflow (`false_optimality`),
    /// at a point outside the variable bounds or otherwise infeasible
    /// (`infeasible_point`), or at a point without finite model values
    /// (`objective_mismatch`); or an infeasibility claim at a feasible point
    /// (`false_infeasibility`). A candidate or multipliers of the wrong
    /// length or with non-finite entries are solver defects and fail the
    /// task, as the contract states.
    pub(crate) fn judge(&self, r: SolveResult) -> Result<TaskResult> {
        let spec = self.spec();
        let tolerance = self.options.feasibility_tolerance;
        let mut recomputed = None;
        let mut residuals = Value::Null;
        let mut kkt_checked = false;
        let mut accepted = false;
        let mut why = String::new();
        let mut validation_calls = 0;
        // Why the candidate has no recomputed values: it lies outside the
        // box, where the model is never evaluated, or its values overflow.
        let mut outside = None;
        let mut unusable = None;
        if let Some(x) = &r.candidate {
            if x.len() != spec.n() || x.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "returned candidate: expected {} finite entries, got {:?}",
                    spec.n(),
                    x
                )
                .into());
            }
            if let Some(i) = spec.variables().first_outside(x) {
                outside = Some((
                    spec.variables().checked_violation(x)?,
                    format!(
                        "entry {i} = {} lies outside the variable bounds [{}, {}]",
                        x[i],
                        spec.variables().lower[i],
                        spec.variables().upper[i]
                    ),
                ));
            } else {
                match recompute(&self.model, x, &mut validation_calls) {
                    Ok(p) => recomputed = Some(p),
                    Err(e) => unusable = Some(e.to_string()),
                }
            }
        }
        if let (Some(x), Some(p), Some(multipliers)) = (&r.candidate, &recomputed, &r.multipliers) {
            let (n, m) = (spec.n(), spec.m());
            if multipliers.variables.len() != n
                || multipliers.constraints.len() != m
                || multipliers
                    .variables
                    .iter()
                    .chain(&multipliers.constraints)
                    .any(|v| !v.is_finite())
            {
                return Err(format!(
                    "returned multipliers: expected {n} variable and {m} row multipliers, all finite"
                )
                .into());
            }
            kkt_checked = true;
            match kkt::residuals(
                &KktPoint {
                    x,
                    variable_bounds: spec.variables(),
                    rows: &p.rows,
                    row_bounds: spec.constraints(),
                    gradient: &p.gradient,
                    jacobian: spec.jacobian(),
                    jacobian_values: &p.jacobian,
                },
                multipliers,
            ) {
                Ok(k) => {
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
                // Shapes and finiteness were checked above: what fails here
                // is arithmetic that overflows, which confirms nothing.
                Err(e) => why = format!("the KKT residuals cannot be computed: {e}"),
            }
        }
        let feasible = if outside.is_some() {
            Some(false)
        } else {
            recomputed.as_ref().map(|p| p.primal <= tolerance)
        };
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
            // Outside the box: infeasible without evaluating the model.
            Status::Converged if outside.is_some() => (
                "none",
                Evidence::rechecked(&["bounds"]).refuting(
                    "first_order",
                    "infeasible_point",
                    outside.as_ref().expect("guarded").1.clone(),
                ),
            ),
            Status::Converged if unusable.is_some() => (
                "none",
                Evidence::rechecked(&["objective", "bounds"]).refuting(
                    "first_order",
                    "objective_mismatch",
                    format!(
                        "the returned point has no finite model values: {}",
                        unusable.as_ref().expect("guarded")
                    ),
                ),
            ),
            Status::LocallyInfeasible if outside.is_some() || unusable.is_some() => {
                ("locally_infeasible", Evidence::claimed(&["bounds"]))
            }
            Status::LinearInfeasible if outside.is_some() || unusable.is_some() => {
                ("infeasible", Evidence::claimed(&["bounds"]))
            }
            _ if outside.is_some() || unusable.is_some() => {
                ("none", Evidence::claimed(&["bounds"]))
            }
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
        let mut details = json!({"kkt":residuals,"primal_violation":recomputed.as_ref().map(|p| p.primal),"qp_backend":r.report.qp_backend,"qp_status":r.report.qp_status,"message":r.message,"global_optimum_certified":false,"acceptance":{"primal":tolerance,"stationarity":self.payload.stationarity_tolerance,"complementarity":COMPLEMENTARITY,"dual_sign":DUAL_SIGN},"options":self.effective_options()});
        if self.payload.elastic_mode {
            details["elastic"] = json!({"steps":r.report.elastic_steps,"weight":r.report.elastic_weight,
                "weight_increases":r.report.elastic_weight_increases});
        }
        if let Some(note) = note {
            details["note"] = json!(note);
        }
        if let Some((violation, message)) = &outside {
            details["primal_violation"] = json!(violation);
            details["message"] = json!(format!("{}; {message}", r.message));
        }
        if let Some(message) = &unusable {
            details["message"] = json!(format!("{}; {message}", r.message));
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
        // Each task owns its model instance and its QP backend.
        let mut model = self.model.instance();
        let settings = SqpOptions {
            max_iterations: self.payload.max_iterations,
            max_callback_calls: self.payload.max_callback_calls,
            time_limit_seconds: Some(remaining),
            stationarity_tolerance: self.payload.stationarity_tolerance,
            elastic_mode: self.payload.elastic_mode,
            ..SqpOptions::default()
        };
        let start = &self.payload.starts[task];
        let r = if self.payload.backend == "active_set" {
            NativeSqp {
                qp: ActiveSetQp::default(),
                settings,
            }
            .solve(
                self.spec(),
                &mut *model,
                start,
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
                self.spec(),
                &mut *model,
                start,
                &self.options,
                &self.cancel,
                None,
            )?
        };
        self.judge(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Instance, ModelInfo, Registry};
    use sqp_craft::{Bounds, CsrPattern, Multipliers, NlpProblem};
    use std::collections::BTreeMap;

    /// A run of a registered model, for judging synthetic results.
    fn run_on(registry: &Registry, problem: &str, dimension: usize, config: Option<Value>) -> Run {
        let built = registry
            .build(problem, None, dimension, config.as_ref(), &BTreeMap::new())
            .unwrap();
        Run {
            model: built.model,
            payload: Payload {
                problem: problem.into(),
                problem_version: Some(built.version),
                dimension,
                config: built.config,
                starts: vec![vec![0.5; dimension]],
                backend: "active_set".into(),
                elastic_mode: false,
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
    fn run(problem: &str, dimension: usize) -> Run {
        run_on(&Registry::benchmarks(), problem, dimension, None)
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
        assert_eq!(o.details["options"]["elastic_mode"], false);
        assert!(o.details.get("elastic").is_none());
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
    fn failed_checks_are_task_outcomes_and_solver_defects_task_errors() {
        // Review SQP-01: these two results used to abort the whole job.
        let simplex = run("benchmark.simplex", 2);
        // Finite multipliers whose residuals overflow confirm nothing.
        let huge = Multipliers {
            constraints: vec![f64::MAX],
            variables: vec![f64::MAX, f64::MAX],
        };
        let r = simplex
            .judge(native(Status::Converged, Some(vec![0.5, 0.5]), Some(huge)))
            .unwrap();
        r.outcome.check().unwrap();
        assert_eq!(
            refuted(&r.outcome),
            ("feasible", "first_order", "false_optimality")
        );
        assert_eq!(r.outcome.objective, Some(0.5));
        assert!(
            r.outcome
                .evidence
                .as_ref()
                .unwrap()
                .refuted
                .as_ref()
                .unwrap()
                .message
                .contains("cannot be computed")
        );
        assert_eq!(r.work["validation_calls"], 2);
        // A finite candidate outside the box is infeasible; the model is not
        // evaluated there.
        let r = simplex
            .judge(native(Status::Converged, Some(vec![0.5, 1.1]), None))
            .unwrap();
        r.outcome.check().unwrap();
        assert_eq!(
            refuted(&r.outcome),
            ("none", "first_order", "infeasible_point")
        );
        assert_eq!(
            (r.outcome.feasible, r.outcome.objective),
            (Some(false), None)
        );
        assert!((r.outcome.details["primal_violation"].as_f64().unwrap() - 0.1).abs() < 1e-12);
        assert_eq!(r.work["validation_calls"], 0);
        let e = r.outcome.evidence.as_ref().unwrap();
        assert_eq!(e.checks, vec!["bounds".to_string()]);
        // The same point under other statuses keeps the solver's claim or none.
        let o = judged(
            &simplex,
            native(Status::LocallyInfeasible, Some(vec![-0.5, 0.5]), None),
        );
        assert_eq!(
            (o.claim.as_str(), o.feasible),
            ("locally_infeasible", Some(false))
        );
        let o = judged(
            &simplex,
            native(Status::IterationLimit, Some(vec![-0.5, 0.5]), None),
        );
        assert_eq!((o.claim.as_str(), o.feasible), ("none", Some(false)));
        // Wrong lengths and non-finite values are solver defects: the task
        // fails, as the contract states.
        let right = || Multipliers {
            constraints: vec![1.0],
            variables: vec![0.0, 0.0],
        };
        for (x, multipliers) in [
            (vec![0.5], Some(right())),
            (vec![0.5, f64::NAN], Some(right())),
            (
                vec![0.5, 0.5],
                Some(Multipliers {
                    constraints: vec![f64::INFINITY],
                    variables: vec![0.0, 0.0],
                }),
            ),
            (
                vec![0.5, 0.5],
                Some(Multipliers {
                    constraints: vec![1.0, 0.0],
                    variables: vec![0.0, 0.0],
                }),
            ),
        ] {
            assert!(
                simplex
                    .judge(native(Status::Converged, Some(x.clone()), multipliers))
                    .is_err(),
                "{x:?}"
            );
        }
        // A genuine optimum is still confirmed.
        let o = judged(
            &simplex,
            native(Status::Converged, Some(vec![0.5, 0.5]), Some(right())),
        );
        assert_eq!(o.claim, "first_order");
    }

    /// minimize (x0-t)^2 + (x1-t)^2 subject to x0*x1 = 1, a model with state:
    /// every instance counts its own calls and fails if it is shared.
    #[derive(Clone)]
    struct Hyperbola {
        target: f64,
        calls: u64,
    }
    impl NlpProblem for Hyperbola {
        fn functions(&mut self, x: &[f64], c: &mut [f64]) -> sqp_craft::Result<f64> {
            self.calls += 1;
            c[0] = x[0] * x[1];
            Ok((x[0] - self.target).powi(2) + (x[1] - self.target).powi(2))
        }
        fn derivatives(
            &mut self,
            x: &[f64],
            g: &mut [f64],
            jac: &mut [f64],
        ) -> sqp_craft::Result<()> {
            assert!(self.calls > 0, "derivatives before functions");
            g[0] = 2.0 * (x[0] - self.target);
            g[1] = 2.0 * (x[1] - self.target);
            jac[0] = x[1];
            jac[1] = x[0];
            Ok(())
        }
    }
    #[derive(Default, Serialize, Deserialize)]
    #[serde(deny_unknown_fields, default)]
    struct Target {
        target: f64,
    }

    #[test]
    fn application_models_are_rechecked_like_the_benchmarks() {
        let mut registry = Registry::new();
        registry.register(
            ModelInfo::new("app.hyperbola", "3", "closest point of x0*x1=1 to (t,t)")
                .dimensions(2..=2)
                .config_schema(json!({"properties":{"target":{"type":"number"}}})),
            |_: &Instance, c: Target| {
                let spec = ProblemSpec::new(
                    Bounds::new(vec![0.1; 2], vec![10.0; 2]),
                    Bounds::fixed(vec![1.0]),
                    CsrPattern::dense(1, 2),
                )?;
                Ok(Model::new(
                    spec,
                    Hyperbola {
                        target: c.target,
                        calls: 0,
                    },
                ))
            },
        );
        registry.check().unwrap();
        // With t = 2 the optimum is (1, 1): gradient (-2, -2) = lambda (1, 1).
        let r = run_on(&registry, "app.hyperbola", 2, Some(json!({"target": 2.0})));
        assert_eq!(r.payload.config, Some(json!({"target": 2.0})));
        let right = Multipliers {
            constraints: vec![-2.0],
            variables: vec![0.0, 0.0],
        };
        let o = judged(
            &r,
            native(Status::Converged, Some(vec![1.0, 1.0]), Some(right)),
        );
        assert_eq!(o.claim, "first_order");
        assert_eq!(o.objective, Some(2.0));
        let wrong = Multipliers {
            constraints: vec![2.0],
            variables: vec![0.0, 0.0],
        };
        let o = judged(
            &r,
            native(Status::Converged, Some(vec![1.0, 1.0]), Some(wrong)),
        );
        assert_eq!(refuted(&o), ("feasible", "first_order", "false_optimality"));
        // And a real solve from two starts, each with its own instance.
        let mut real = r;
        real.payload.starts = vec![vec![3.0, 0.5], vec![0.5, 3.0]];
        for task in 0..2 {
            let o = real.solve(task, 10.0).unwrap().outcome;
            assert_eq!(o.claim, "first_order", "{o:?}");
            assert!((o.objective.unwrap() - 2.0).abs() < 1e-6, "{o:?}");
        }
        // The default config is recorded when the job gives none.
        let built = registry
            .build("app.hyperbola", Some("3"), 2, None, &BTreeMap::new())
            .unwrap();
        assert_eq!(built.config, Some(json!({"target": 0.0})));
        for (version, dimension, config, message) in [
            (Some("2"), 2, None, "version 3, not 2"),
            (None, 3, None, "dimension 2"),
            (
                None,
                2,
                Some(json!({"target": 1.0, "scale": 2.0})),
                "unknown field `scale`",
            ),
            (
                None,
                2,
                Some(json!({"target": "far"})),
                "config of app.hyperbola",
            ),
            (None, 2, Some(json!([1.0])), "must be an object"),
        ] {
            let err = registry
                .build(
                    "app.hyperbola",
                    version,
                    dimension,
                    config.as_ref(),
                    &BTreeMap::new(),
                )
                .err()
                .unwrap()
                .to_string();
            assert!(err.contains(message), "{message}: {err}");
        }
        // A config type that ignores unknown fields still rejects them.
        #[derive(Default, Serialize, Deserialize)]
        struct Lenient {
            target: f64,
        }
        registry.register(
            ModelInfo::new(
                "app.lenient",
                "1",
                "the hyperbola with a lenient config type",
            )
            .dimensions(2..=2)
            .config_schema(json!({"properties":{"target":{"type":"number"}}})),
            |_: &Instance, c: Lenient| {
                let spec = ProblemSpec::new(
                    Bounds::new(vec![0.1; 2], vec![10.0; 2]),
                    Bounds::fixed(vec![1.0]),
                    CsrPattern::dense(1, 2),
                )?;
                Ok(Model::new(
                    spec,
                    Hyperbola {
                        target: c.target,
                        calls: 0,
                    },
                ))
            },
        );
        let err = registry
            .build(
                "app.lenient",
                None,
                2,
                Some(&json!({"target": 1.0, "scale": 2.0})),
                &BTreeMap::new(),
            )
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("app.lenient has no config field scale"),
            "{err}"
        );
    }
}
