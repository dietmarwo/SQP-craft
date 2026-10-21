//! The models a worker solves, by problem identifier.
//!
//! A job names a registered model; it never carries code. The stock worker
//! registers three benchmarks ([`Registry::benchmarks`]); an application
//! worker registers its own smooth models and runs the same adapter
//! (`examples/custom_worker.rs`, `docs/job-api/models.md`):
//!
//! ```
//! use serde::{Deserialize, Serialize};
//! use serde_json::json;
//! use sqp_craft::{Bounds, CsrPattern, NlpProblem, ProblemSpec};
//! use sqp_job_worker::registry::{Instance, Model, ModelInfo, Registry};
//!
//! /// minimize sum((x[i] - shift)^2)
//! #[derive(Clone)]
//! struct Shifted(f64);
//! impl NlpProblem for Shifted {
//!     fn functions(&mut self, x: &[f64], _: &mut [f64]) -> sqp_craft::Result<f64> {
//!         Ok(x.iter().map(|v| (v - self.0).powi(2)).sum())
//!     }
//!     fn derivatives(&mut self, x: &[f64], g: &mut [f64], _: &mut [f64]) -> sqp_craft::Result<()> {
//!         for (g, v) in g.iter_mut().zip(x) {
//!             *g = 2.0 * (v - self.0);
//!         }
//!         Ok(())
//!     }
//! }
//! #[derive(Default, Serialize, Deserialize)]
//! #[serde(deny_unknown_fields, default)]
//! struct Config {
//!     shift: f64,
//! }
//!
//! let mut registry = Registry::benchmarks();
//! registry.register(
//!     ModelInfo::new("app.shifted", "1", "sum of (x[i]-shift)^2 on [-5,5]")
//!         .dimensions(1..=256)
//!         .config_schema(json!({"properties": {"shift": {"type": "number"}}})),
//!     |model: &Instance, config: Config| {
//!         let n = model.dimension();
//!         let spec = ProblemSpec::new(
//!             Bounds::new(vec![-5.0; n], vec![5.0; n]),
//!             Bounds::free(0),
//!             CsrPattern::dense(0, n),
//!         )?;
//!         Ok(Model::new(spec, Shifted(config.shift)))
//!     },
//! );
//! registry.check().unwrap(); // derivatives against finite differences
//! // optimization_job_api::worker::run(sqp_job_worker::Adapter::new(registry));
//! ```
//!
//! A factory receives an [`Instance`] (the job's dimension and its input
//! bytes, already checked against their SHA-256) and the typed `config`. It
//! returns a [`Model`]: the validated [`ProblemSpec`] and a way to make one
//! fresh [`NlpProblem`] per task. Every task, and every independent recheck
//! of a result, gets its own instance, so a model may keep mutable state
//! (a propagation cache, say) without locks. The worker records the version
//! and the effective config, defaults included, in the resolved job.
//! Factories run at validation and again at execution, so they must be
//! deterministic and quick.
use optimization_job_api::Result;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use sqp_craft::{
    NlpProblem, ProblemSpec,
    derivative_check::{self, DerivativeOptions, DerivativeReport},
};
use std::{collections::BTreeMap, ops::RangeInclusive, sync::Arc};

/// Largest dimension of any model: the native SQP's dense limit.
pub const MAX_DIMENSION: usize = 256;

/// The config of a model without parameters: `config` may be omitted or
/// `{}`, and any field is an error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoConfig {}

type Maker = Arc<dyn Fn() -> Box<dyn NlpProblem> + Send + Sync>;

/// A built model: its problem spec and a maker of fresh model instances.
#[derive(Clone)]
pub struct Model {
    spec: ProblemSpec,
    make: Maker,
}

impl Model {
    /// A model whose instances are clones of `prototype`. Put large
    /// read-only data (a parsed input) in an `Arc` to share it between tasks.
    pub fn new<M>(spec: ProblemSpec, prototype: M) -> Self
    where
        M: NlpProblem + Clone + Send + Sync + 'static,
    {
        Self::from_fn(spec, move || prototype.clone())
    }
    /// A model whose instances `make` creates, for a model that is not
    /// `Clone` or whose per-task state is built from shared data.
    pub fn from_fn<M, F>(spec: ProblemSpec, make: F) -> Self
    where
        M: NlpProblem + 'static,
        F: Fn() -> M + Send + Sync + 'static,
    {
        Self {
            spec,
            make: Arc::new(move || Box::new(make()) as Box<dyn NlpProblem>),
        }
    }
    pub fn spec(&self) -> &ProblemSpec {
        &self.spec
    }
    /// A fresh model instance, as every task and every recheck gets one.
    pub fn instance(&self) -> Box<dyn NlpProblem> {
        (self.make)()
    }
    /// Compares the gradient and the Jacobian of a fresh instance with
    /// finite differences at `x` (inside the bounds), with the default
    /// [`DerivativeOptions`]; `passed` says whether they agree.
    pub fn check_derivatives(&self, x: &[f64]) -> Result<DerivativeReport> {
        let mut model = self.instance();
        Ok(derivative_check::check_derivatives(
            &self.spec,
            &mut *model,
            x,
            &DerivativeOptions::default(),
        )?)
    }
}

/// A data input a model reads, with a small example for `describe`.
#[derive(Clone, Debug)]
pub struct InputInfo {
    pub name: String,
    pub description: String,
    /// File name of the example input.
    pub example_file: String,
    /// UTF-8 content of the example input (for example a few CSV rows).
    pub example_text: String,
}

/// What `describe` says about a model, and how its examples are built.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    id: String,
    version: String,
    description: String,
    dimensions: RangeInclusive<usize>,
    config_schema: Option<Value>,
    inputs: Vec<InputInfo>,
    example_dimension: Option<usize>,
    example_start: Option<Vec<f64>>,
    check_points: Vec<Vec<f64>>,
    about: Map<String, Value>,
}

impl ModelInfo {
    /// A model `id` at `version`, for every dimension 1..=256, without
    /// parameters and inputs.
    pub fn new(id: &str, version: &str, description: &str) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            description: description.into(),
            dimensions: 1..=MAX_DIMENSION,
            config_schema: None,
            inputs: Vec::new(),
            example_dimension: None,
            example_start: None,
            check_points: Vec::new(),
            about: Map::new(),
        }
    }
    /// The dimensions a job may give (within 1..=256).
    pub fn dimensions(mut self, range: RangeInclusive<usize>) -> Self {
        self.dimensions = range;
        self
    }
    /// The JSON Schema of `config`: an object schema with `properties`, one
    /// per field of the config type. The registry adds `type: object`,
    /// `additionalProperties: false` and each field's default.
    pub fn config_schema(mut self, schema: Value) -> Self {
        self.config_schema = Some(schema);
        self
    }
    /// Declares the data input `name`. Jobs on this model must name exactly
    /// the declared inputs. `example_file` and `example_text` form the input
    /// of the model's `describe` examples.
    pub fn input(
        mut self,
        name: &str,
        description: &str,
        example_file: &str,
        example_text: &str,
    ) -> Self {
        self.inputs.push(InputInfo {
            name: name.into(),
            description: description.into(),
            example_file: example_file.into(),
            example_text: example_text.into(),
        });
        self
    }
    /// The dimension of the examples (default: the smallest allowed).
    pub fn example_dimension(mut self, dimension: usize) -> Self {
        self.example_dimension = Some(dimension);
        self
    }
    /// The start of the examples (default: the middle of the box, or a
    /// point one unit inside a one-sided bound, or zero).
    pub fn example_start(mut self, start: Vec<f64>) -> Self {
        self.example_start = Some(start);
        self
    }
    /// Another point (of the example dimension) where [`Registry::check`]
    /// compares the derivatives with finite differences.
    pub fn check_point(mut self, x: Vec<f64>) -> Self {
        self.check_points.push(x);
        self
    }
    /// Adds `key` to the model's `describe` entry, for example `bounds` or
    /// the meaning of the variables.
    pub fn about(mut self, key: &str, value: Value) -> Self {
        self.about.insert(key.into(), value);
        self
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn inputs(&self) -> &[InputInfo] {
        &self.inputs
    }
    pub fn dimension_range(&self) -> &RangeInclusive<usize> {
        &self.dimensions
    }
}

/// What a factory knows about the job it builds a model for.
pub struct Instance<'a> {
    id: &'a str,
    dimension: usize,
    inputs: &'a BTreeMap<String, Vec<u8>>,
}

impl<'a> Instance<'a> {
    /// The problem identifier.
    pub fn id(&self) -> &str {
        self.id
    }
    /// The job's dimension, inside the declared range.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// The bytes of the declared input `name`, checked against the SHA-256
    /// the job gives.
    pub fn input(&self, name: &str) -> Result<&'a [u8]> {
        self.inputs
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("{} declares no input {name}", self.id).into())
    }
    /// The input `name` as UTF-8 text.
    pub fn input_text(&self, name: &str) -> Result<&'a str> {
        std::str::from_utf8(self.input(name)?)
            .map_err(|e| format!("input {name} of {}: {e}", self.id).into())
    }
}

/// A model built for one job: the version and effective config the worker
/// records in the resolved job, and the model.
pub struct Built {
    pub version: String,
    /// The effective config, defaults included; `None` for a model without
    /// parameters whose job gave no `config`.
    pub config: Option<Value>,
    pub model: Model,
}

type Build = Box<dyn Fn(&Instance, Option<&Value>) -> Result<(Value, Model)>>;

struct Entry {
    info: ModelInfo,
    /// The serialized default config: an object, empty without parameters.
    default_config: Value,
    build: Build,
}

impl Entry {
    fn has_parameters(&self) -> bool {
        self.default_config
            .as_object()
            .is_some_and(|o| !o.is_empty())
    }
    /// The config schema of `describe`; models without parameters accept `{}`.
    fn schema(&self) -> Value {
        self.info.config_schema.clone().unwrap_or_else(
            || json!({"type":"object","additionalProperties":false,"maxProperties":0}),
        )
    }
}

/// An example job of one model: dimension, start and inputs.
pub(crate) struct Example {
    pub id: String,
    pub version: String,
    pub dimension: usize,
    pub config: Option<Value>,
    pub start: Vec<f64>,
    pub inputs: Vec<InputInfo>,
}

/// Registered models by problem identifier, in registration order.
#[derive(Default)]
pub struct Registry {
    entries: Vec<Entry>,
}

impl Registry {
    /// A registry without models.
    pub fn new() -> Self {
        Self::default()
    }
    /// The stock benchmark models (`benchmark.quadratic`,
    /// `benchmark.simplex`, `benchmark.rosenbrock`).
    pub fn benchmarks() -> Self {
        let mut registry = Self::new();
        crate::benchmarks::register(&mut registry);
        registry
    }

    /// Registers a model. `C` is its typed config: `NoConfig` for a model
    /// without parameters, else a `Default` serde type whose fields
    /// [`ModelInfo::config_schema`] lists (use
    /// `#[serde(deny_unknown_fields, default)]`). A config field the type
    /// does not serialize is rejected anyway.
    ///
    /// # Panics
    ///
    /// On a programming error: an empty id or version, an id registered
    /// twice, a dimension range outside 1..=256, an input declared twice, a
    /// config type that does not serialize to an object, or a config schema
    /// that does not list exactly the config's fields.
    pub fn register<C, F>(&mut self, mut info: ModelInfo, factory: F) -> &mut Self
    where
        C: Serialize + DeserializeOwned + Default + 'static,
        F: Fn(&Instance, C) -> Result<Model> + 'static,
    {
        let id = info.id.clone();
        assert!(
            !id.is_empty() && !info.version.is_empty(),
            "a model needs an identifier and a version"
        );
        assert!(self.entry(&id).is_none(), "model {id} registered twice");
        let (lo, hi) = (*info.dimensions.start(), *info.dimensions.end());
        assert!(
            1 <= lo && lo <= hi && hi <= MAX_DIMENSION,
            "the dimensions of {id} must lie in 1..=256"
        );
        let default_config = serde_json::to_value(C::default())
            .unwrap_or_else(|e| panic!("the default config of {id} does not serialize: {e}"));
        let fields = default_config
            .as_object()
            .unwrap_or_else(|| panic!("the config of {id} must serialize to an object"))
            .clone();
        if let Some(schema) = &mut info.config_schema {
            let object = schema
                .as_object_mut()
                .unwrap_or_else(|| panic!("the config schema of {id} must be an object"));
            object.entry("type").or_insert(json!("object"));
            object.entry("additionalProperties").or_insert(json!(false));
            let properties = object
                .get_mut("properties")
                .and_then(Value::as_object_mut)
                .unwrap_or_else(|| panic!("the config schema of {id} needs properties"));
            assert!(
                properties.keys().collect::<std::collections::BTreeSet<_>>()
                    == fields.keys().collect(),
                "the config schema of {id} lists {:?}, its config type serializes {:?}",
                properties.keys().collect::<Vec<_>>(),
                fields.keys().collect::<Vec<_>>()
            );
            for (name, default) in &fields {
                if let Some(p) = properties.get_mut(name).and_then(Value::as_object_mut) {
                    p.entry("default").or_insert(default.clone());
                }
            }
        } else {
            assert!(
                fields.is_empty(),
                "model {id} has config fields but no config schema"
            );
        }
        let names = info.inputs.iter().map(|i| &i.name);
        assert!(
            names
                .clone()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == info.inputs.len(),
            "model {id} declares an input twice"
        );
        let label = id.clone();
        let build: Build = Box::new(move |instance, given| {
            let config: C = match given {
                None => C::default(),
                Some(value) => {
                    let object = value
                        .as_object()
                        .ok_or_else(|| format!("config of {label} must be an object"))?;
                    let config: C = serde_json::from_value(value.clone())
                        .map_err(|e| format!("config of {label}: {e}"))?;
                    // A field the type ignores would be silently dropped.
                    let back = serde_json::to_value(&config)?;
                    if let Some(k) = object.keys().find(|k| back.get(k.as_str()).is_none()) {
                        return Err(format!("{label} has no config field {k}").into());
                    }
                    config
                }
            };
            let effective = serde_json::to_value(&config)?;
            Ok((effective, factory(instance, config)?))
        });
        self.entries.push(Entry {
            info,
            default_config,
            build,
        });
        self
    }

    fn entry(&self, id: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.info.id == id)
    }
    /// The registered models, in registration order.
    pub fn models(&self) -> impl Iterator<Item = &ModelInfo> {
        self.entries.iter().map(|e| &e.info)
    }
    /// The model `id`, if registered.
    pub fn info(&self, id: &str) -> Option<&ModelInfo> {
        self.entry(id).map(|e| &e.info)
    }

    /// Builds model `id` as a job states it: `version` (the registered one
    /// when `None`), `dimension`, `config` (the defaults when `None`) and
    /// the bytes of exactly the declared inputs.
    pub fn build(
        &self,
        id: &str,
        version: Option<&str>,
        dimension: usize,
        config: Option<&Value>,
        inputs: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Built> {
        let entry = self.entry(id).ok_or_else(|| {
            format!(
                "unknown problem {id}; registered: {}",
                self.entries
                    .iter()
                    .map(|e| e.info.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let info = &entry.info;
        if let Some(v) = version.filter(|v| *v != info.version) {
            return Err(format!("problem {id} has version {}, not {v}", info.version).into());
        }
        if !info.dimensions.contains(&dimension) {
            let (lo, hi) = (info.dimensions.start(), info.dimensions.end());
            return Err(if lo == hi {
                format!("{id} needs dimension {lo}")
            } else {
                format!("{id} needs a dimension in {lo}..={hi}")
            }
            .into());
        }
        for declared in &info.inputs {
            if !inputs.contains_key(&declared.name) {
                return Err(format!("{id} needs the job input {}", declared.name).into());
            }
        }
        if let Some(extra) = inputs
            .keys()
            .find(|k| !info.inputs.iter().any(|i| &i.name == *k))
        {
            return Err(if info.inputs.is_empty() {
                format!("{id} reads no inputs; remove input {extra}")
            } else {
                format!("{id} does not read input {extra}")
            }
            .into());
        }
        let instance = Instance {
            id,
            dimension,
            inputs,
        };
        let (effective, model) = (entry.build)(&instance, config)?;
        if model.spec.n() != dimension {
            return Err(format!(
                "the factory of {id} built {} variables for dimension {dimension}",
                model.spec.n()
            )
            .into());
        }
        Ok(Built {
            version: info.version.clone(),
            config: (config.is_some() || entry.has_parameters()).then_some(effective),
            model,
        })
    }

    /// The example of model `id`: built with the default config and the
    /// example inputs, with its start inside the bounds.
    pub(crate) fn example(&self, id: &str) -> Result<(Example, Model)> {
        let entry = self.entry(id).ok_or("unknown problem")?;
        let info = &entry.info;
        let dimension = info.example_dimension.unwrap_or(*info.dimensions.start());
        let inputs = info
            .inputs
            .iter()
            .map(|i| (i.name.clone(), i.example_text.clone().into_bytes()))
            .collect();
        let built = self.build(id, None, dimension, None, &inputs)?;
        let spec = built.model.spec();
        let start = match &info.example_start {
            Some(x) => x.clone(),
            None => {
                let v = spec.variables();
                v.lower
                    .iter()
                    .zip(&v.upper)
                    .map(|(&l, &u)| match (l.is_finite(), u.is_finite()) {
                        (true, true) => 0.5 * (l + u),
                        (true, false) => l + 1.0,
                        (false, true) => u - 1.0,
                        (false, false) => 0.0,
                    })
                    .collect()
            }
        };
        spec.check_point(&start, "example start")?;
        Ok((
            Example {
                id: id.into(),
                version: built.version,
                dimension,
                config: built.config,
                start,
                inputs: info.inputs.clone(),
            },
            built.model,
        ))
    }

    /// Checks every model before it is advertised: its example builds with
    /// the default config and the example inputs, its start lies inside the
    /// bounds, and the gradient and the Jacobian agree with finite
    /// differences at the example start and at every
    /// [`ModelInfo::check_point`]. Call it in a test of every worker.
    pub fn check(&self) -> Result<()> {
        let mut failures = Vec::new();
        for entry in &self.entries {
            let id = &entry.info.id;
            let (example, model) = match self.example(id) {
                Ok(built) => built,
                Err(e) => {
                    failures.push(format!("{id}: the example does not build: {e}"));
                    continue;
                }
            };
            for x in std::iter::once(&example.start).chain(&entry.info.check_points) {
                match model.check_derivatives(x) {
                    Ok(report) if report.passed => (),
                    Ok(report) => failures.push(format!(
                        "{id}: derivatives disagree with finite differences at {x:?}: {:?}",
                        report.issues
                    )),
                    Err(e) => failures.push(format!("{id}: derivative check at {x:?}: {e}")),
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; ").into())
        }
    }

    /// The `problems` entries of `describe`, and the examples that build.
    pub(crate) fn discovery(&self, profiles: &[&str]) -> (Value, Vec<Example>) {
        let mut problems = Vec::new();
        let mut examples = Vec::new();
        for entry in &self.entries {
            let info = &entry.info;
            let (lo, hi) = (info.dimensions.start(), info.dimensions.end());
            let mut p = json!({"id":info.id,"version":info.version,"description":info.description,
                "dimension":format!("{lo}..{hi}"),"profiles":profiles});
            for (k, v) in &info.about {
                p[k] = v.clone();
            }
            if entry.has_parameters() {
                p["config"] = entry.schema();
            }
            if !info.inputs.is_empty() {
                p["inputs"] = info
                    .inputs
                    .iter()
                    .map(|i| (i.name.clone(), json!(i.description)))
                    .collect::<Map<_, _>>()
                    .into();
            }
            match self.example(&info.id) {
                Ok((example, model)) => {
                    if p.get("bounds").is_none() {
                        let v = model.spec().variables();
                        p["bounds"] = json!({"lower":v.lower,"upper":v.upper});
                    }
                    examples.push(example);
                }
                Err(e) => p["example_error"] = json!(e.to_string()),
            }
            problems.push(p);
        }
        (problems.into(), examples)
    }

    /// Per-model rules of the payload schemas (version, dimensions, config)
    /// and the `$defs` they refer to: one config schema per model with
    /// parameters, and `no_config` for all others.
    pub(crate) fn schema_rules(&self) -> (Vec<Value>, Map<String, Value>) {
        let mut defs = Map::new();
        defs.insert(
            "no_config".into(),
            json!({"type":"object","additionalProperties":false,"maxProperties":0}),
        );
        let rules = self
            .entries
            .iter()
            .map(|e| {
                let info = &e.info;
                let config = if e.has_parameters() {
                    let name = format!("config.{}", info.id);
                    // A JSON Pointer escapes "~" and "/".
                    let pointer = name.replace('~', "~0").replace('/', "~1");
                    defs.insert(name, e.schema());
                    format!("#/$defs/{pointer}")
                } else {
                    "#/$defs/no_config".into()
                };
                json!({"if":{"properties":{"problem":{"const":info.id}},"required":["problem"]},
                    "then":{"properties":{"problem_version":{"enum":[info.version, null]},
                        "dimension":{"minimum":info.dimensions.start(),"maximum":info.dimensions.end()},
                        "config":{"anyOf":[{"$ref":config},{"type":"null"}]}}}})
            })
            .collect();
        (rules, defs)
    }
}
