# Application models

A job never carries code. It names a registered model, and the worker
evaluates that model natively: it rechecks every answer with its own model
instance before reporting it, a job can be re-run from its record, and
nothing deserializes a function. The stock `sqp-job-worker` registers three
benchmarks. Your own smooth NLP runs in an *application worker*: a small
binary that registers its models in a `Registry` and runs the same adapter,
with both profiles (`sqp.solve/1`, `sqp.evaluate/1`), both QP backends,
elastic mode and the independent KKT recheck.

## A worker of your own

```toml
[dependencies]
sqp-job-worker = { git = "https://github.com/dietmarwo/SQP-craft" }
sqp-craft = { git = "https://github.com/dietmarwo/SQP-craft" }
# The revision SQP-craft's root Cargo.toml pins; another one would be a
# second, incompatible copy of the crate.
optimization-job-api = { git = "https://github.com/dietmarwo/optimization-jobs", rev = "70e210fbf562900c1be8f278e2ab2852ef5934e3" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

```rust
use sqp_job_worker::{Adapter, registry::{Instance, Model, ModelInfo, Registry}};

fn main() {
    let mut registry = Registry::benchmarks(); // or Registry::new()
    registry.register(
        ModelInfo::new("app.my-model", "1", "what it minimizes, in one line")
            .dimensions(3..=3)
            .config_schema(serde_json::json!({"properties": {"weight": {"type": "number"}}}))
            .input("data", "CSV rows t,y", "data.csv", "t,y\n0,1\n1,2\n2,4\n3,8\n"),
        |model: &Instance, config: MyConfig| {
            let rows = parse(model.input_text("data")?)?;
            Ok(Model::new(my_spec()?, MyNlp::new(rows, config.weight)))
        },
    );
    optimization_job_api::worker::run(Adapter::new(registry));
}
```

`MyNlp` implements `sqp_craft::NlpProblem` (`functions` and `derivatives`,
see the [API](../api.md)); `MyConfig` is a `Default` serde type with
`#[serde(deny_unknown_fields, default)]`, or `NoConfig` for a model without
parameters. Build the binary and register it with the runner under a worker
name of its own:

```sh
cargo build --release
optimization-run --root ROOT workers add --id my-app --executable target/release/my-app
optimization-run --root ROOT problems describe my-app
```

`crates/sqp-job-worker/examples/custom_worker.rs` is a complete worker: a
least-squares fit of `a*exp(-b*t)+c` to a hashed CSV input, constrained to
pass through the row at `config.anchor_time`. The
[tutorial](../../tutorials/job-api/README.md) runs it (`decay-fit.json`) and
rechecks its answer in Python.

## Registering a model

| `ModelInfo` method | Says |
| --- | --- |
| `new(id, version, description)` | identifier, version and one line for `describe` |
| `dimensions(lo..=hi)` | the dimensions a job may give, within 1..=256 (default all) |
| `config_schema(json)` | the JSON Schema `properties` of the config type; the registry adds `type`, `additionalProperties: false` and the defaults |
| `input(name, description, example_file, example_text)` | a data input, with a small example for `describe` |
| `example_dimension(n)`, `example_start(x)` | the example job (default: smallest dimension, middle of the box) |
| `check_point(x)` | another point for `Registry::check` |
| `about(key, value)` | any other key of the `describe` entry (`bounds`, variable names) |

A job states `problem` (the id), `dimension`, and optionally
`problem_version` (it must equal the registered version) and `config`. The
resolved job records the version and the effective config, defaults
included, so the record shows exactly what ran.

## What a factory receives

The factory gets an `Instance` and the typed config, and returns a `Model`:
the validated `ProblemSpec` (bounds, rows, Jacobian pattern) and a way to make
model instances.

| `Instance` method | Returns |
| --- | --- |
| `id()`, `dimension()` | the problem id and the job's dimension |
| `input(name)`, `input_text(name)` | the bytes (or UTF-8 text) of a declared input, checked against its SHA-256 |

`Model::new(spec, prototype)` clones the prototype for every task;
`Model::from_fn(spec, make)` calls `make` instead. Each task, and each
recheck of a result, gets its own instance, so tasks that run in parallel
never share mutable model state. Put large read-only data, such as the parsed
input, in an `Arc`. The instance itself need not be `Send`.

## Inputs

Data reaches a model through the envelope's `inputs`, never inline:

```json
"inputs": [{"name": "data", "file": "decay.csv", "sha256": "b779f43d..."}],
"payload": {"problem": "app.decay-fit", "dimension": 3, "config": {"anchor_time": 1.0}, ...}
```

`file` is relative to the job file. The runner checks the SHA-256 and copies
the file into the attempt, so a re-run reads the same bytes; inputs are at
most 64 MiB each and 16 per job. A job must name exactly the inputs its model
declares; the stock benchmarks declare none. Parse an input once in the
factory.

## What the worker checks

At validation, before a job is accepted: the id is registered and the
version matches; the dimension is in range; the config deserializes and has
no unknown field; the inputs are exactly the declared ones and match their
hashes; the factory succeeds (it rejects malformed data with its own message)
and builds a spec of the job's dimension; every start lies inside the bounds;
`elastic_mode` comes with `backend: "active_set"`. At execution, every
`converged` result is rechecked with a fresh instance: the objective,
bounds, constraints and the KKT residuals with the solver's multipliers.
Only a result that passes is claimed `first_order`.

## Testing a model

- `registry.check()` builds every model's example and compares its gradient
  and Jacobian with finite differences at the example start and at every
  `check_point`. Call it in a test: a wrong derivative fails there, before
  any job.
- `model.check_derivatives(x)` does the same for one built model at `x`.
- `Worker::prepare` resolves a job exactly as `validate` does, and
  `Prepared::solve` runs one task; `optimization_job_api::worker::execute`
  runs a whole job in a directory.

`crates/sqp-job-worker/tests/custom_worker.rs` drives the example model
through these, and `tests/discovery.rs` turns every `describe` example into a
job and runs it.

## Discovery

Each model's `describe` entry lists `id`, `version`, `description`,
`dimension`, `profiles`, `bounds`, the `config` schema (models with
parameters) and the `inputs`. The worker also lists `payload_schemas` (a
JSON Schema per profile, with per-model rules for version, dimensions and
config) and `examples` (one job per model and profile, in resolved form, with
the example inputs as `content_text`). Keep the output small: the runner's
`workers add` reads at most 1 MiB of `describe` (64 KiB before
optimization-jobs `069d87c`; stay under that to register with older
runners).

## Rules for models

- **Smooth and deterministic.** SQP needs twice continuously differentiable
  functions inside the bounds, and the recheck must reproduce the solver's
  values.
- **Inside the box when solving, anywhere when evaluating.** `sqp.solve/1`
  never evaluates the model outside the variable bounds. `sqp.evaluate/1`
  evaluates the given points as they are, outside the bounds too. Return
  `Error::UndefinedPoint` (or another error) wherever the model is
  undefined, inside or outside the box: a solve then shortens its trial
  step, and an evaluation reports claim `none` with the message instead of
  failing the job.
- **Quick factories.** A factory runs at validation, which has a 30-second
  limit, and again at execution.
