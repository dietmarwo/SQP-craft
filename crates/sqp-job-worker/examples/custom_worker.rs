//! An application worker: the stock benchmarks plus a data-backed model of
//! its own, a constrained least-squares fit of `y = a*exp(-b*t) + c` to the
//! rows `t,y` of a CSV file:
//!
//! ```text
//! minimize    0.5 * sum_i (a*exp(-b*t_i) + c - y_i)^2
//! subject to  a*exp(-b*T) + c = Y      (the curve passes through the row t = T)
//!             0 <= a <= 10,  0 <= b <= 5,  -5 <= c <= 5
//! ```
//!
//! `T` is the model parameter `config.anchor_time` (default 0, the first
//! measurement). The job names the model, its config and its data input; the
//! data reaches the model as SHA-256-checked bytes, and the worker records
//! the version and the effective config in the resolved job:
//!
//! ```json
//! "payload": {"problem": "app.decay-fit", "dimension": 3, "config": {"anchor_time": 1.0},
//!             "starts": [[5.0, 2.5, 0.0]], "backend": "active_set", "elastic_mode": true, ...},
//! "inputs": [{"name": "data", "file": "decay.csv", "sha256": "..."}]
//! ```
//!
//! Build it, check its derivatives against finite differences, and register
//! it with the runner like the stock worker:
//!
//! ```text
//! cargo build --release --locked -p sqp-job-worker --example custom_worker
//! target/release/examples/custom_worker check
//! optimization-run workers add --id sqp-curve-fit --executable target/release/examples/custom_worker
//! ```
//!
//! `tutorials/job-api/run.py` runs it on `decay.csv` (`decay-fit.json`), and
//! `tests/custom_worker.rs` tests the model through the worker.
use optimization_job_api::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqp_craft::{Bounds, CsrPattern, NlpProblem, ProblemSpec};
use sqp_job_worker::{
    Adapter,
    registry::{Instance, Model, ModelInfo, Registry},
};
use std::sync::Arc;

/// The tutorial's data: `3*exp(-0.7*t) + 0.5` plus noise, t = 0, 0.25, ..., 5.
pub const EXAMPLE_DATA: &str = include_str!("../../../tutorials/job-api/decay.csv");

/// The parameters of `app.decay-fit`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FitConfig {
    /// `t` of the data row the fitted curve must pass through.
    pub anchor_time: f64,
}
impl Default for FitConfig {
    fn default() -> Self {
        Self { anchor_time: 0.0 }
    }
}

/// `(t, y)` rows of a CSV file; a first line that is not numeric is a header.
pub fn samples(text: &str) -> Result<Vec<(f64, f64)>> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let parsed = match fields.as_slice() {
            [t, y] => t.parse::<f64>().ok().zip(y.parse::<f64>().ok()),
            _ => None,
        };
        match parsed {
            Some((t, y)) if t.is_finite() && y.is_finite() => rows.push((t, y)),
            None if i == 0 => (), // header
            _ => {
                return Err(format!("data line {}: expected two finite numbers t,y", i + 1).into());
            }
        }
    }
    if rows.len() < 4 {
        return Err("data needs at least four rows".into());
    }
    Ok(rows)
}

/// One task's model instance. The parsed rows are shared between tasks.
#[derive(Clone)]
struct DecayFit {
    rows: Arc<Vec<(f64, f64)>>,
    anchor_time: f64,
}

impl NlpProblem for DecayFit {
    fn functions(&mut self, x: &[f64], c: &mut [f64]) -> sqp_craft::Result<f64> {
        let (a, b, offset) = (x[0], x[1], x[2]);
        c[0] = a * (-b * self.anchor_time).exp() + offset;
        Ok(0.5
            * self
                .rows
                .iter()
                .map(|(t, y)| (a * (-b * t).exp() + offset - y).powi(2))
                .sum::<f64>())
    }
    fn derivatives(&mut self, x: &[f64], g: &mut [f64], jac: &mut [f64]) -> sqp_craft::Result<()> {
        let (a, b, offset) = (x[0], x[1], x[2]);
        g.fill(0.0);
        for (t, y) in self.rows.iter() {
            let e = (-b * t).exp();
            let r = a * e + offset - y;
            g[0] += r * e;
            g[1] -= r * a * t * e;
            g[2] += r;
        }
        let e = (-b * self.anchor_time).exp();
        jac.copy_from_slice(&[e, -a * self.anchor_time * e, 1.0]);
        Ok(())
    }
}

/// The factory: parses the hashed input, finds the anchor row, builds the spec.
fn decay_fit(model: &Instance, config: FitConfig) -> Result<Model> {
    let rows = samples(model.input_text("data")?)?;
    let anchor = rows
        .iter()
        .find(|(t, _)| *t == config.anchor_time)
        .ok_or_else(|| format!("no data row has t = {}", config.anchor_time))?
        .1;
    let spec = ProblemSpec::new(
        Bounds::new(vec![0.0, 0.0, -5.0], vec![10.0, 5.0, 5.0]),
        Bounds::fixed(vec![anchor]),
        CsrPattern::dense(1, 3),
    )?;
    Ok(Model::new(
        spec,
        DecayFit {
            rows: Arc::new(rows),
            anchor_time: config.anchor_time,
        },
    ))
}

/// Registers the application's model.
pub fn register(registry: &mut Registry) {
    registry.register(
        ModelInfo::new(
            "app.decay-fit",
            "1",
            "least-squares fit of a*exp(-b*t)+c to CSV rows t,y; the curve passes through the row t = config.anchor_time",
        )
        .dimensions(3..=3)
        .about("variables", json!(["a", "b", "c"]))
        .config_schema(json!({"properties":{"anchor_time":{"type":"number",
            "description":"t of the data row the curve must pass through"}}}))
        .input(
            "data",
            "CSV rows t,y (at least four), an optional header line",
            "decay.csv",
            EXAMPLE_DATA,
        )
        .check_point(vec![2.0, 1.0, 0.3]),
        decay_fit,
    );
}

/// The benchmarks plus the application's model.
pub fn registry() -> Registry {
    let mut registry = Registry::benchmarks();
    register(&mut registry);
    registry
}

fn main() {
    // `check`: every model's example and derivatives, before registering
    // the worker; the rest is the worker protocol (describe, validate, execute).
    if std::env::args().skip(1).eq(["check"]) {
        match registry().check() {
            Ok(()) => println!("all models pass their checks"),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        return;
    }
    optimization_job_api::worker::run(Adapter::new(registry()));
}
