# Summary

- [Introduction](introduction.md)
- [Project overview](README.md)

# Using the solver

- [API contract](docs/api.md)
- [Rust crate: problem contract](crates/core/README.md)
- [Rust crate: solver](crates/sqp/README.md)
- [Rust crate: Clarabel backend](crates/clarabel/README.md)
- [Rust crate: active-set backend](crates/activeset/README.md)
- [Rust crate: parallel helpers](crates/parallel/README.md)
- [Python package](python/README.md)
- [Dependencies](docs/dependencies.md)

# Methods and limits

- [Design notes](docs/design-notes.md)
- [Structured SQP with sparse curvature](docs/structured-sqp.md)
- [Elastic mode](docs/elastic-mode.md)
- [Second-order correction](docs/second-order-correction.md)
- [Dense active-set QP backend](docs/active-set-qp.md)
- [Gaps compared with mature SQP codes](docs/gaps.md)
- [Benchmark results](docs/benchmark-results.md)

# Tutorials

- [Application tutorials](tutorials/README.md)
  - [SQP1: Constrained calibration](tutorials/constrained-calibration/README.md)
  - [SQP2: Multistart, units and QP backends](tutorials/multistart-scaling/README.md)
  - [SQP3: Hanging chain](tutorials/hanging-chain/README.md)
  - [SQP4: Optimal control by collocation](tutorials/collocation/README.md)
  - [SQP5: Earth-Mars rendezvous](tutorials/low-thrust-pykep/README.md)
  - [SQP6: Reaction kinetics](tutorials/reaction-kinetics/README.md)
  - [X2: Power dispatch](tutorials/power-dispatch/README.md)

# Jobs and agents

- [AI context for modeling](ai-context.md)
- [Shared job API](docs/job-api.md)
  - [Native SQP jobs and multistart batches](tutorials/job-api/README.md)
  - [Application models](docs/job-api/models.md)
  - [Job API validation](docs/job-api/VALIDATION.md)

# Project

- [Changelog](CHANGELOG.md)
