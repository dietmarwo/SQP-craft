//! The discovery convention: every profile has a payload schema and at least
//! one example, and every example is a job that validates and runs, for the
//! stock worker, for the custom worker of `examples/custom_worker.rs` and for
//! a worker whose registry has no benchmark at all.
mod common;
#[allow(dead_code)]
#[path = "../examples/custom_worker.rs"]
mod custom_worker;

use common::TempDir;
use optimization_job_api::{
    JobSpec,
    storage::digest,
    worker::{Worker, execute},
};
use serde_json::{Value, json};
use sqp_craft::{Bounds, CsrPattern, NlpProblem, ProblemSpec};
use sqp_job_worker::{
    Adapter,
    registry::{Instance, Model, ModelInfo, NoConfig, Registry},
};
use std::fs;

/// minimize sum((x[i]-1)^2) subject to x[0] - x[1] >= -1, on a box that is
/// open on some sides: the example start comes from the bounds.
#[derive(Clone)]
struct OpenBox;
impl NlpProblem for OpenBox {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> sqp_craft::Result<f64> {
        c[0] = x[0] - x[1];
        Ok(x.iter().map(|v| (v - 1.0).powi(2)).sum())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> sqp_craft::Result<()> {
        for (g, v) in g.iter_mut().zip(x) {
            *g = 2.0 * (v - 1.0);
        }
        jac.copy_from_slice(&[1.0, -1.0, 0.0]);
        Ok(())
    }
}

/// The application models only: no `benchmark.*`.
fn without_benchmarks() -> Registry {
    let mut registry = Registry::new();
    custom_worker::register(&mut registry);
    registry.register(
        ModelInfo::new("app.open-box", "2", "a box with infinite sides").dimensions(3..=3),
        |_: &Instance, _: NoConfig| {
            let spec = ProblemSpec::new(
                Bounds::new(
                    vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
                    vec![f64::INFINITY, 4.0, f64::INFINITY],
                ),
                Bounds::new(vec![-1.0], vec![f64::INFINITY]),
                CsrPattern::dense(1, 3),
            )?;
            Ok(Model::new(spec, OpenBox))
        },
    );
    registry
}

fn workers() -> Vec<(&'static str, Adapter)> {
    vec![
        ("stock", Adapter::default()),
        ("custom", Adapter::new(custom_worker::registry())),
        ("without-benchmarks", Adapter::new(without_benchmarks())),
    ]
}

#[test]
fn every_profile_has_a_payload_schema_and_an_example() {
    for (name, adapter) in workers() {
        adapter.registry().check().unwrap();
        let d = adapter.describe();
        let bytes = serde_json::to_vec(&d).unwrap().len();
        // optimization-run's `workers add` reads at most 64 KiB of describe.
        assert!(bytes < 48 * 1024, "{name}: describe has {bytes} bytes");
        let profiles = d["profiles"].as_array().unwrap();
        assert_eq!(profiles.len(), 2);
        for profile in profiles {
            let profile = profile.as_str().unwrap();
            let schema = &d["payload_schemas"][profile];
            assert_eq!(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
            assert_eq!(
                schema["$id"],
                format!("urn:optimization-jobs:payload:{profile}")
            );
            // Unknown fields are errors, at the top and in every config.
            assert_eq!(schema["additionalProperties"], false);
            assert!(schema["$comment"].is_string());
            let defs = schema["$defs"].as_object().unwrap();
            for (name, def) in defs {
                if name != "vector" {
                    assert_eq!(def["additionalProperties"], false, "{name}");
                }
            }
            // Every model rule refers to a config schema in $defs.
            for rule in schema["allOf"].as_array().unwrap() {
                if let Some(r) = rule["then"]["properties"]["config"]["$ref"].as_str() {
                    let name = r.strip_prefix("#/$defs/").unwrap().replace("~1", "/");
                    assert!(defs.contains_key(&name.replace("~0", "~")), "{r}");
                }
            }
            let examples = d["examples"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["profile"] == profile)
                .count();
            assert!(examples >= 1, "{name}: no example of {profile}");
        }
        let models = d["problems"].as_array().unwrap();
        for model in models {
            assert!(model.get("example_error").is_none(), "{name}: {model}");
            assert_eq!(model["profiles"], json!(profiles));
        }
        // One example per model and profile.
        assert_eq!(d["examples"].as_array().unwrap().len(), 2 * models.len());
        for example in d["examples"].as_array().unwrap() {
            for input in example["inputs"].as_array().unwrap() {
                // No profile reads a fixed-format input: no input_schemas.
                assert!(input.get("format").is_none() && d.get("input_schemas").is_none());
                assert!(input["content_text"].is_string());
            }
        }
    }
}

#[test]
fn the_fit_advertises_its_config_and_inputs() {
    let d = Adapter::new(custom_worker::registry()).describe();
    let fit = d["problems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == "app.decay-fit")
        .unwrap();
    assert_eq!(fit["version"], "1");
    assert_eq!(fit["dimension"], "3..3");
    assert_eq!(
        fit["config"]["properties"]["anchor_time"]["default"],
        json!(0.0)
    );
    assert_eq!(fit["config"]["additionalProperties"], false);
    assert!(fit["inputs"]["data"].is_string());
    // Bounds from the built example model.
    assert_eq!(fit["bounds"]["lower"], json!([0.0, 0.0, -5.0]));
    let example = d["examples"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["payload"]["problem"] == "app.decay-fit")
        .unwrap();
    assert_eq!(example["payload"]["config"], json!({"anchor_time": 0.0}));
    assert_eq!(example["inputs"][0]["name"], "data");
    // The middle of the box.
    assert_eq!(example["payload"]["starts"], json!([[5.0, 2.5, 0.0]]));
}

/// Turns a `describe` example into a job, as a client would: inputs written
/// with their SHA-256, a request key, the worker and the payload schema.
fn job(dir: &std::path::Path, example: &Value) -> JobSpec {
    let mut inputs = Vec::new();
    for input in example["inputs"].as_array().unwrap() {
        let text = input["content_text"].as_str().unwrap();
        let file = input["file"].as_str().unwrap();
        fs::write(dir.join(file), text).unwrap();
        inputs.push(json!({"name":input["name"],"file":file,"sha256":digest(text.as_bytes())}));
    }
    let s: JobSpec = serde_json::from_value(json!({"schema_version":"optimization.job/0.1",
        "request_key":"example","worker":"sqp","payload_schema":example["profile"],
        "payload":example["payload"],"resources":example["resources"],"limits":example["limits"],
        "inputs":inputs}))
    .unwrap();
    s.validate().unwrap();
    s
}

#[test]
fn every_example_is_a_job_that_validates_and_runs() {
    for (name, adapter) in workers() {
        let d = adapter.describe();
        for example in d["examples"].as_array().unwrap() {
            let dir = TempDir::new("example");
            let mut s = job(dir.path(), example);
            let prepared = adapter
                .prepare(&mut s, dir.path())
                .unwrap_or_else(|e| panic!("{name}: {example}: {e}"));
            // Examples are in resolved form: validation changes nothing.
            assert_eq!(s.payload, example["payload"], "{name}");
            assert!(prepared.tasks() >= 1);
            s.task_count = Some(prepared.tasks());
            fs::write(
                dir.path().join("resolved.json"),
                serde_json::to_vec(&s).unwrap(),
            )
            .unwrap();
            execute(&adapter, dir.path()).unwrap_or_else(|e| panic!("{name}: {example}: {e}"));
            let run: Value =
                serde_json::from_slice(&fs::read(dir.path().join("run.json")).unwrap()).unwrap();
            assert_eq!(run["status"], "completed", "{name}: {run}");
            assert_eq!(run["completed_tasks"], s.task_count.unwrap());
            let outcome = &run["results"][0]["outcome"];
            if example["profile"] == sqp_job_worker::PROFILE {
                // Every example solve reaches a rechecked first-order point.
                assert_eq!(outcome["claim"], "first_order", "{name}: {outcome}");
            } else {
                assert_eq!(outcome["termination"], "evaluated");
            }
        }
    }
}
