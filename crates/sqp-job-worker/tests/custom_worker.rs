//! The application model of `examples/custom_worker.rs` through the worker:
//! the derivative check, hashed inputs, typed config, and solves through
//! `prepare` and `solve` with the independent KKT recheck.
mod common;
#[allow(dead_code)]
#[path = "../examples/custom_worker.rs"]
mod custom_worker;

use common::{TempDir, job, one_slot};
use optimization_job_api::worker::{Prepared, Worker};
use serde_json::{Value, json};
use sqp_craft::{Bounds, CsrPattern, NlpProblem, ProblemSpec};
use sqp_job_worker::{
    Adapter, PROFILE,
    registry::{Instance, Model, ModelInfo, NoConfig, Registry},
};
use std::collections::BTreeMap;

const DATA: &str = custom_worker::EXAMPLE_DATA;

fn fit_payload(config: Value, starts: Value) -> Value {
    json!({"problem":"app.decay-fit","dimension":3,"config":config,"starts":starts,
        "backend":"active_set","elastic_mode":true,"max_iterations":200,"max_callback_calls":4000,
        "feasibility_tolerance":1e-7,"stationarity_tolerance":1e-6})
}

/// The fit's loss and constraint value at `x`, computed here from the CSV.
fn independent(x: &[f64], anchor_time: f64) -> (f64, f64) {
    let rows = custom_worker::samples(DATA).unwrap();
    let curve = |t: f64| x[0] * (-x[1] * t).exp() + x[2];
    let loss = 0.5
        * rows
            .iter()
            .map(|(t, y)| (curve(*t) - y).powi(2))
            .sum::<f64>();
    let anchor = rows.iter().find(|(t, _)| *t == anchor_time).unwrap().1;
    (loss, curve(anchor_time) - anchor)
}

#[test]
fn the_registry_passes_its_derivative_checks() {
    custom_worker::registry().check().unwrap();
    // The nonlinear row of a later anchor, at a point the check does not use.
    let inputs = BTreeMap::from([("data".to_string(), DATA.as_bytes().to_vec())]);
    let built = custom_worker::registry()
        .build(
            "app.decay-fit",
            Some("1"),
            3,
            Some(&json!({"anchor_time": 1.0})),
            &inputs,
        )
        .unwrap();
    for x in [[3.0, 0.7, 0.5], [9.0, 4.0, -2.0], [0.5, 0.1, 1.0]] {
        let report = built.model.check_derivatives(&x).unwrap();
        assert!(report.passed, "{x:?}: {report:?}");
    }
}

/// `(x0-1)^2 + x0*x1^2` with the wrong sign of the derivative in `x1`.
#[derive(Clone)]
struct WrongGradient;
impl NlpProblem for WrongGradient {
    fn functions(&mut self, x: &[f64], _: &mut [f64]) -> sqp_craft::Result<f64> {
        Ok((x[0] - 1.0).powi(2) + x[0] * x[1] * x[1])
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> sqp_craft::Result<()> {
        g[0] = 2.0 * (x[0] - 1.0) + x[1] * x[1];
        g[1] = -2.0 * x[0] * x[1];
        Ok(())
    }
}

#[test]
fn a_wrong_gradient_fails_the_check_before_any_job() {
    let mut registry = Registry::new();
    registry.register(
        ModelInfo::new("app.wrong", "1", "a model with a wrong gradient")
            .dimensions(2..=2)
            .example_start(vec![0.5, 0.5]),
        |_: &Instance, _: NoConfig| {
            let spec = ProblemSpec::new(
                Bounds::new(vec![-2.0; 2], vec![2.0; 2]),
                Bounds::free(0),
                CsrPattern::dense(0, 2),
            )?;
            Ok(Model::new(spec, WrongGradient))
        },
    );
    let err = registry.check().unwrap_err().to_string();
    assert!(
        err.contains("app.wrong") && err.contains("disagree"),
        "{err}"
    );
}

#[test]
fn the_fit_is_solved_and_rechecked_through_the_worker() {
    let adapter = Adapter::new(custom_worker::registry());
    let dir = TempDir::new("fit");
    let starts = json!([[5.0, 2.5, 0.0], [1.0, 0.2, 2.0], [9.0, 4.5, -4.0]]);
    let mut s = job(
        dir.path(),
        PROFILE,
        fit_payload(json!({"anchor_time": 1.0}), starts),
        json!({"cpu_slots":3,"outer_workers":3,"inner_threads":1}),
        &[("data", "decay.csv", DATA)],
    );
    let prepared: Box<dyn Prepared> = adapter.prepare(&mut s, dir.path()).unwrap();
    assert_eq!(prepared.tasks(), 3);
    // The resolved job records the version and the effective config.
    assert_eq!(s.payload["problem_version"], "1");
    assert_eq!(s.payload["config"], json!({"anchor_time": 1.0}));
    let mut objectives = Vec::new();
    for task in 0..3 {
        let r = prepared.solve(task, 20.0).unwrap();
        let o = r.outcome;
        o.check().unwrap();
        assert_eq!(
            (o.termination.as_str(), o.claim.as_str()),
            ("converged", "first_order"),
            "{o:?}"
        );
        let e = o.evidence.as_ref().unwrap();
        assert_eq!(e.level, "rechecked");
        assert_eq!(e.checks, ["objective", "bounds", "constraints", "kkt"]);
        assert!(o.details["kkt"]["stationarity"].as_f64().unwrap() <= 1e-6);
        assert_eq!(o.details["options"]["elastic_mode"], true);
        // The recheck used fresh instances: two more model calls.
        assert_eq!(r.work["validation_calls"], 2);
        let x = o.x.as_ref().unwrap();
        let (loss, row) = independent(x, 1.0);
        assert!((o.objective.unwrap() - loss).abs() <= 1e-12 * loss.max(1.0));
        assert!(row.abs() <= 1e-7, "{row}");
        // Noise of 0.01 on 21 rows: close to the generating parameters.
        assert!(
            (x[0] - 3.0).abs() < 0.05 && (x[1] - 0.7).abs() < 0.03 && (x[2] - 0.5).abs() < 0.03
        );
        objectives.push(o.objective.unwrap());
    }
    // One basin: every start finds the same point.
    assert!(objectives.iter().all(|f| (f - objectives[0]).abs() < 1e-9));
    // Without a config, the default anchor (t = 0) is used and recorded.
    let mut s = job(
        dir.path(),
        PROFILE,
        fit_payload(json!({}), json!([[5.0, 2.5, 0.0]])),
        one_slot(),
        &[("data", "decay.csv", DATA)],
    );
    s.payload.as_object_mut().unwrap().remove("config");
    let prepared = adapter.prepare(&mut s, dir.path()).unwrap();
    assert_eq!(s.payload["config"], json!({"anchor_time": 0.0}));
    let o = prepared.solve(0, 20.0).unwrap().outcome;
    assert_eq!(o.claim, "first_order");
    let (_, row) = independent(o.x.as_ref().unwrap(), 0.0);
    assert!(row.abs() <= 1e-7);
}

#[test]
fn inputs_are_accepted_exactly_when_the_model_declares_them() {
    let adapter = Adapter::new(custom_worker::registry());
    let dir = TempDir::new("inputs");
    let fit = |config: Value| fit_payload(config, json!([[5.0, 2.5, 0.0]]));
    let data = [("data", "decay.csv", DATA)];
    let error = |mut s: optimization_job_api::JobSpec| {
        adapter
            .prepare(&mut s, dir.path())
            .err()
            .expect("rejected")
            .to_string()
    };
    for (payload, inputs, message) in [
        (fit(json!({})), &[][..], "needs the job input data"),
        (
            fit(json!({})),
            &[("data", "decay.csv", DATA), ("extra", "extra.csv", "t,y\n")][..],
            "does not read input extra",
        ),
        (
            fit(json!({"anchor_time": 0.3})),
            &data[..],
            "no data row has t = 0.3",
        ),
        (
            fit(json!({"anchor": 0.0})),
            &data[..],
            "config of app.decay-fit",
        ),
        (
            json!({"problem":"benchmark.simplex","dimension":2,"starts":[[0.5,0.5]],"backend":"active_set",
                "max_iterations":10,"max_callback_calls":100,"feasibility_tolerance":1e-7,"stationarity_tolerance":1e-6}),
            &data[..],
            "reads no inputs",
        ),
    ] {
        let err = error(job(dir.path(), PROFILE, payload, one_slot(), inputs));
        assert!(err.contains(message), "{message}: {err}");
    }
    let mut s = job(dir.path(), PROFILE, fit(json!({})), one_slot(), &data);
    s.payload["problem_version"] = json!("2");
    assert!(error(s).contains("version 1, not 2"));
    let mut s = job(dir.path(), PROFILE, fit(json!({})), one_slot(), &data);
    s.payload["dimension"] = json!(2);
    assert!(error(s).contains("needs dimension 3"));
    // A file that does not match its hash never reaches the factory.
    let mut s = job(dir.path(), PROFILE, fit(json!({})), one_slot(), &data);
    std::fs::write(dir.path().join("decay.csv"), "t,y\n0,1\n").unwrap();
    assert!(error(s.clone()).contains("SHA-256"));
    std::fs::write(dir.path().join("decay.csv"), DATA).unwrap();
    adapter.prepare(&mut s, dir.path()).unwrap();
    // Malformed data is the factory's error.
    let s = job(
        dir.path(),
        PROFILE,
        fit(json!({})),
        one_slot(),
        &[("data", "bad.csv", "t,y\n0,1\n1,x\n")],
    );
    assert!(error(s).contains("data line 3"));
}
