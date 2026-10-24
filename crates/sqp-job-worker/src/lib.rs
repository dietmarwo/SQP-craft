//! SQP-craft adapters for the common job protocol (`optimization.job/0.1`).
//!
//! - `sqp.solve/1`: one native SQP solve per start on a registered smooth
//!   model, with the active-set or the Clarabel QP backend (optionally in
//!   elastic mode), and an independent KKT recheck of every result;
//! - `sqp.evaluate/1`: the function values of given points, without solving.
//!
//! Models come from a [`Registry`]: the stock benchmarks and whatever an
//! application registers ([`registry`] module docs,
//! `examples/custom_worker.rs`, `docs/job-api/models.md`). Callbacks never
//! cross the job boundary: a job names a model, its version, its config and
//! its SHA-256-checked data inputs. Every task gets its own model instance
//! and QP backend.
//!
//! An application worker is a binary of a few lines:
//!
//! ```no_run
//! let registry = sqp_job_worker::Registry::benchmarks(); // plus its own models
//! optimization_job_api::worker::run(sqp_job_worker::Adapter::new(registry));
//! ```
#![forbid(unsafe_code)]
mod benchmarks;
mod discovery;
mod evaluate;
pub mod registry;
mod solve;

pub use evaluate::PROFILE as EVALUATE_PROFILE;
pub use registry::Registry;
pub use solve::PROFILE;

use optimization_job_api::{
    JobSpec, Result, checked_input,
    worker::{self, Prepared, Worker},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

/// The engine version: this crate shares the workspace version with
/// `sqp-craft-solver`, so the reported version cannot drift from the linked
/// solver.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
/// All profiles accepted by this worker.
pub const PROFILES: [&str; 2] = [PROFILE, EVALUATE_PROFILE];
/// KKT acceptance of `first_order`, besides the payload's tolerances.
pub const COMPLEMENTARITY: f64 = 1e-6;
pub const DUAL_SIGN: f64 = 1e-8;
/// Properties recomputed when a returned point is re-evaluated.
const POINT_CHECKS: [&str; 3] = ["objective", "bounds", "constraints"];

/// Largest `inner_threads` of `sqp.solve/1`: `Options::threads` allows 1..=8.
pub const MAX_INNER_THREADS: usize = 8;

/// The SQP worker: both profiles on the models of a [`Registry`]. An
/// application worker registers its own models and runs
/// `optimization_job_api::worker::run(Adapter::new(registry))`.
pub struct Adapter {
    registry: Registry,
}
impl Adapter {
    pub fn new(registry: Registry) -> Self {
        Self { registry }
    }
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}
/// The stock worker: the three benchmark models.
impl Default for Adapter {
    fn default() -> Self {
        Self::new(Registry::benchmarks())
    }
}

/// The bytes of the job's inputs by name, checked against their SHA-256.
fn read_inputs(spec: &JobSpec, base: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    spec.inputs
        .iter()
        .map(|i| {
            checked_input(base, i)
                .map(|bytes| (i.name.clone(), bytes))
                .map_err(|e| format!("input {}: {e}", i.name).into())
        })
        .collect()
}

impl Worker for Adapter {
    fn describe(&self) -> Value {
        let (problems, examples) = self.registry.discovery(&PROFILES);
        let mut d = worker::description(
            "sqp-craft",
            ENGINE_VERSION,
            json!(PROFILES),
            problems,
            MAX_INNER_THREADS,
        );
        d["backends"] = json!(solve::BACKENDS);
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
        d["inputs"] = json!(
            "exactly the data inputs the model declares (problems[].inputs), checked against their SHA-256; the stock benchmarks read none"
        );
        d["payload_schemas"] = discovery::payload_schemas(&self.registry);
        d["examples"] = discovery::examples(&examples);
        d
    }
    fn prepare(&self, s: &mut JobSpec, base: &Path) -> Result<Box<dyn Prepared>> {
        let profile = s.payload_schema.as_str();
        if !PROFILES.contains(&profile) {
            return Err(format!(
                "unsupported payload schema; this worker accepts {}",
                PROFILES.join(", ")
            )
            .into());
        }
        let inner = s.resources.inner_threads;
        if profile == EVALUATE_PROFILE && inner != 1 {
            return Err(
                format!("{EVALUATE_PROFILE} evaluates serially; use inner_threads=1").into(),
            );
        }
        if inner > MAX_INNER_THREADS {
            return Err(format!(
                "{PROFILE} supports inner_threads=1..{MAX_INNER_THREADS}; {EVALUATE_PROFILE} evaluates points"
            )
            .into());
        }
        let inputs = read_inputs(s, base)?;
        if profile == EVALUATE_PROFILE {
            let p: evaluate::Payload = serde_json::from_value(s.payload.clone())?;
            let built = self.registry.build(
                &p.problem,
                p.problem_version.as_deref(),
                p.dimension,
                p.config.as_ref(),
                &inputs,
            )?;
            let (resolved, prepared) = evaluate::prepare(p, built)?;
            s.payload = serde_json::to_value(&resolved)?;
            return Ok(Box::new(prepared));
        }
        let p: solve::Payload = serde_json::from_value(s.payload.clone())?;
        let built = self.registry.build(
            &p.problem,
            p.problem_version.as_deref(),
            p.dimension,
            p.config.as_ref(),
            &inputs,
        )?;
        let (resolved, prepared) = solve::prepare(p, built, inner)?;
        s.payload = serde_json::to_value(&resolved)?;
        Ok(Box::new(prepared))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimization_job_api::result::Outcome;

    fn job(profile: &str, payload: Value, inner_threads: usize) -> JobSpec {
        serde_json::from_value(json!({"schema_version":"optimization.job/0.1",
            "request_key":"test","worker":"sqp","payload_schema":profile,"payload":payload,
            "resources":{"cpu_slots":inner_threads,"outer_workers":1,"inner_threads":inner_threads},
            "limits":{"wall_time_seconds":10.0,"grace_seconds":1.0},"inputs":[]}))
        .unwrap()
    }
    fn simplex(extra: Value) -> Value {
        let mut p = json!({"problem":"benchmark.simplex","dimension":4,"starts":[[0.1,0.2,0.3,0.4]],
            "backend":"active_set","max_iterations":200,"max_callback_calls":4000,
            "feasibility_tolerance":1e-7,"stationarity_tolerance":1e-6});
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        p
    }
    fn error(s: &mut JobSpec) -> String {
        Adapter::default()
            .prepare(s, Path::new("."))
            .err()
            .unwrap()
            .to_string()
    }

    #[test]
    fn evaluate_checks_points_inside_and_outside_the_bounds() {
        let mut s = job(
            EVALUATE_PROFILE,
            json!({"problem":"benchmark.simplex","dimension":2,"points":[[0.25,0.75],[0.5,0.25],[2.0,-1.0],[1e200,0.0]],
                "feasibility_tolerance":1e-7}),
            1,
        );
        let p = Adapter::default().prepare(&mut s, Path::new(".")).unwrap();
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
        // The resolved payload records the model version.
        assert_eq!(s.payload["problem_version"], "1");
        assert!(s.payload.get("config").is_none());
        for (key, value, message) in [
            ("points", json!([]), "1..1000000"),
            ("points", json!([[1.0]]), "point 0"),
            ("feasibility_tolerance", json!(0.0), "tolerance"),
            ("problem_version", json!("2"), "version 1, not 2"),
            ("config", json!({"scale":1}), "unknown field `scale`"),
        ] {
            let mut bad = s.clone();
            bad.payload[key] = value;
            let err = error(&mut bad);
            assert!(err.contains(message), "{message}: {err}");
        }
        let mut bad = s.clone();
        bad.resources =
            serde_json::from_value(json!({"cpu_slots":2,"outer_workers":1,"inner_threads":2}))
                .unwrap();
        assert!(error(&mut bad).contains("inner_threads=1"));
    }

    #[test]
    fn old_payloads_resolve_with_the_version_and_elastic_mode_off() {
        let mut s = job(PROFILE, simplex(json!({})), 1);
        let p = Adapter::default().prepare(&mut s, Path::new(".")).unwrap();
        assert_eq!(p.tasks(), 1);
        let mut expected = simplex(json!({"problem_version":"1","elastic_mode":false}));
        assert_eq!(s.payload, expected);
        // Preparing the resolved job again changes nothing (the runner checks).
        let mut again = s.clone();
        Adapter::default()
            .prepare(&mut again, Path::new("."))
            .unwrap();
        assert_eq!(again, s);
        // An explicit empty config of a model without parameters is kept.
        let mut s = job(PROFILE, simplex(json!({"config":{}})), 1);
        Adapter::default().prepare(&mut s, Path::new(".")).unwrap();
        expected["config"] = json!({});
        assert_eq!(s.payload, expected);
    }

    #[test]
    fn elastic_mode_needs_the_active_set_backend_and_is_reported() {
        let mut s = job(PROFILE, simplex(json!({"elastic_mode":true})), 1);
        let p = Adapter::default().prepare(&mut s, Path::new(".")).unwrap();
        assert_eq!(s.payload["elastic_mode"], true);
        let o = p.solve(0, 10.0).unwrap().outcome;
        assert_eq!(o.claim, "first_order", "{o:?}");
        assert!((o.objective.unwrap() - 0.25).abs() < 1e-8);
        assert_eq!(o.details["options"]["elastic_mode"], true);
        assert_eq!(o.details["options"]["backend"], "active_set");
        assert!(o.details["elastic"]["steps"].is_u64());
        let mut s = job(
            PROFILE,
            simplex(json!({"elastic_mode":true,"backend":"clarabel"})),
            1,
        );
        assert!(error(&mut s).contains("elastic_mode needs backend active_set"));
        let mut s = job(PROFILE, simplex(json!({"elastic_mode":"yes"})), 1);
        assert!(error(&mut s).contains("expected a boolean"));
    }

    #[test]
    fn the_stock_models_reject_inputs_and_unknown_fields() {
        let dir = std::env::temp_dir().join(format!("sqp-worker-inputs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("data.csv"), "t,y\n0,1\n").unwrap();
        let mut s = job(PROFILE, simplex(json!({})), 1);
        s.inputs = serde_json::from_value(json!([{"name":"data","file":"data.csv",
            "sha256":optimization_job_api::storage::digest(b"t,y\n0,1\n")}]))
        .unwrap();
        let err = Adapter::default()
            .prepare(&mut s, &dir)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("reads no inputs"), "{err}");
        s.inputs[0].sha256 = "0".repeat(64);
        let err = Adapter::default()
            .prepare(&mut s, &dir)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("SHA-256"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
        for extra in [
            json!({"ignored_option":true}),
            json!({"problem":"benchmark.unknown"}),
            json!({"dimension":0}),
            json!({"dimension":300}),
        ] {
            let mut s = job(PROFILE, simplex(extra.clone()), 1);
            assert!(
                Adapter::default().prepare(&mut s, Path::new(".")).is_err(),
                "{extra}"
            );
        }
        let mut s = job(
            PROFILE,
            simplex(json!({"problem":"benchmark.rosenbrock","dimension":1,"starts":[[0.0]]})),
            1,
        );
        assert!(error(&mut s).contains("2..=256"));
    }

    #[test]
    fn description_reports_the_workspace_engine_version_and_profiles() {
        let d = Adapter::default().describe();
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
        // The model entries keep their earlier keys.
        let problems = d["problems"].as_array().unwrap();
        assert_eq!(
            problems.iter().map(|p| &p["id"]).collect::<Vec<_>>(),
            [
                "benchmark.quadratic",
                "benchmark.simplex",
                "benchmark.rosenbrock"
            ]
        );
        assert_eq!(problems[0]["dimension"], "1..256");
        assert_eq!(problems[0]["bounds"], json!([-5, 5]));
        assert_eq!(problems[1]["constraint"], "sum(x)=1");
        assert_eq!(problems[2]["dimension"], "2..256");
        assert!(problems.iter().all(|p| p.get("config").is_none()));
    }

    #[test]
    fn registered_derivatives_match_finite_differences() {
        Registry::benchmarks().check().unwrap();
        let registry = Registry::benchmarks();
        for id in [
            "benchmark.quadratic",
            "benchmark.simplex",
            "benchmark.rosenbrock",
        ] {
            let built = registry.build(id, None, 4, None, &BTreeMap::new()).unwrap();
            let report = built
                .model
                .check_derivatives(&[0.2, 0.3, 0.1, 0.4])
                .unwrap();
            assert!(report.passed, "{id}: {report:?}");
        }
    }
}
