//! The discovery keys of `describe`: JSON Schemas of the payloads
//! (`payload_schemas`) and runnable examples (`examples`), built from the
//! registry, so an application worker advertises its own models.
use crate::{
    MAX_INNER_THREADS, evaluate,
    registry::{Example, Registry},
    solve,
};
use serde_json::{Value, json};

const DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";

/// The schema properties both profiles share: the model and its config.
/// The two optional fields deserialize as `Option`, so an explicit `null`
/// means the same as leaving the field out; the schemas accept it.
fn model_properties(registry: &Registry) -> Value {
    let ids: Vec<&str> = registry.models().map(|m| m.id()).collect();
    json!({
        "problem": {"type":"string","enum":ids,"description":"registered model id (describe: problems)"},
        "problem_version": {"type":["string","null"],"minLength":1,"description":"default (absent or null): the registered version"},
        "dimension": {"type":"integer","minimum":1,"maximum":crate::registry::MAX_DIMENSION},
        "config": {"type":["object","null"],"description":"model parameters (problems[].config); absent or null: the model's defaults"}
    })
}

/// One task per entry, each a vector of `$defs/vector`.
fn vectors(description: &str) -> Value {
    json!({"type":"array","minItems":1,"maxItems":1_000_000,"description":description,
        "items":{"$ref":"#/$defs/vector"}})
}

/// The `$defs` of both schemas: a point and the models' config schemas.
fn defs(mut configs: serde_json::Map<String, Value>) -> Value {
    configs.insert(
        "vector".into(),
        json!({"type":"array","minItems":1,"maxItems":crate::registry::MAX_DIMENSION,"items":{"type":"number"}}),
    );
    configs.into()
}

/// `payload_schemas`: one strict JSON Schema per profile.
pub(crate) fn payload_schemas(registry: &Registry) -> Value {
    let (rules, configs) = registry.schema_rules();
    let defs = defs(configs);
    let mut solve = json!({
        "$schema": DRAFT,
        "$id": format!("urn:optimization-jobs:payload:{}", solve::PROFILE),
        "$comment": format!("Cross-field rules: every start has `dimension` entries inside the model's variable bounds (a start may violate constraint rows; it is never projected); elastic_mode needs backend active_set; the envelope's inputs are exactly the model's declared inputs (problems[].inputs; none for the benchmarks); resources.inner_threads is 1..{} with outer_workers * inner_threads <= cpu_slots. The resolved job records problem_version, elastic_mode and, for a model with parameters, the effective config.", MAX_INNER_THREADS),
        "type": "object",
        "additionalProperties": false,
        "required": ["problem","dimension","starts","backend","max_iterations","max_callback_calls","feasibility_tolerance","stationarity_tolerance"],
        "properties": model_properties(registry),
        "allOf": rules.clone(),
        "$defs": defs.clone()
    });
    let p = solve["properties"].as_object_mut().expect("object");
    p.insert(
        "starts".into(),
        vectors("one finite start per task, inside the model's bounds"),
    );
    p.insert(
        "backend".into(),
        json!({"enum":solve::BACKENDS,"description":"QP backend: dense active-set or Clarabel interior point"}),
    );
    p.insert(
        "elastic_mode".into(),
        json!({"type":"boolean","default":false,"description":"l1-elastic step QPs instead of feasibility restoration; needs backend active_set"}),
    );
    p.insert(
        "max_iterations".into(),
        json!({"type":"integer","minimum":0,"description":"accepted steps per task; 0 only evaluates the start"}),
    );
    p.insert(
        "max_callback_calls".into(),
        json!({"type":"integer","minimum":1,"description":"function plus derivative calls per task"}),
    );
    p.insert(
        "feasibility_tolerance".into(),
        json!({"type":"number","exclusiveMinimum":0}),
    );
    p.insert(
        "stationarity_tolerance".into(),
        json!({"type":"number","exclusiveMinimum":0}),
    );
    solve["allOf"].as_array_mut().expect("array").push(json!({
        "if":{"properties":{"elastic_mode":{"const":true}},"required":["elastic_mode"]},
        "then":{"properties":{"backend":{"const":"active_set"}}}
    }));
    let mut evaluate = json!({
        "$schema": DRAFT,
        "$id": format!("urn:optimization-jobs:payload:{}", evaluate::PROFILE),
        "$comment": "Cross-field rules: every point has `dimension` finite entries (points may lie outside the bounds); the envelope's inputs are exactly the model's declared inputs; resources.inner_threads must be 1. The resolved job records problem_version and, for a model with parameters, the effective config.",
        "type": "object",
        "additionalProperties": false,
        "required": ["problem","dimension","points","feasibility_tolerance"],
        "properties": model_properties(registry),
        "allOf": rules,
        "$defs": defs
    });
    let p = evaluate["properties"].as_object_mut().expect("object");
    p.insert(
        "points".into(),
        vectors("one point per task, also outside the bounds"),
    );
    p.insert(
        "feasibility_tolerance".into(),
        json!({"type":"number","exclusiveMinimum":0}),
    );
    json!({solve::PROFILE: solve, evaluate::PROFILE: evaluate})
}

fn inputs(example: &Example) -> Value {
    example
        .inputs
        .iter()
        .map(|i| json!({"name":i.name,"file":i.example_file,"content_text":i.example_text}))
        .collect()
}

/// `examples`: for every model whose example builds, one job per profile,
/// in resolved form (the payload `validate` returns), on one CPU slot.
pub(crate) fn examples(examples: &[Example]) -> Value {
    let one = json!({"cpu_slots":1,"outer_workers":1,"inner_threads":1});
    let limits = json!({"wall_time_seconds":30.0,"grace_seconds":1.0});
    let mut out = Vec::new();
    for e in examples {
        let mut solve = json!({"problem":e.id,"problem_version":e.version,"dimension":e.dimension,
            "starts":[e.start],"backend":"active_set","elastic_mode":false,"max_iterations":200,
            "max_callback_calls":4000,"feasibility_tolerance":1e-7,"stationarity_tolerance":1e-6});
        let mut evaluate = json!({"problem":e.id,"problem_version":e.version,"dimension":e.dimension,
            "points":[e.start],"feasibility_tolerance":1e-7});
        if let Some(config) = &e.config {
            solve["config"] = config.clone();
            evaluate["config"] = config.clone();
        }
        let n = e.dimension;
        out.push(json!({"profile":solve::PROFILE,
            "description":format!("{}: one SQP solve in {n} variables from one start, active-set backend", e.id),
            "resources":one,"limits":limits,"payload":solve,"inputs":inputs(e)}));
        out.push(json!({"profile":evaluate::PROFILE,
            "description":format!("{}: the objective, constraint values and violation at one point", e.id),
            "resources":one,"limits":limits,"payload":evaluate,"inputs":inputs(e)}));
    }
    out.into()
}
