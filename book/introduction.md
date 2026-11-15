# SQP-craft documentation

SQP-craft is a sequential quadratic programming solver for smooth constrained
problems with derivatives, written in Rust with Python bindings and a job
worker. This site renders the pages a user needs:

- [Project overview](README.md): what the solver does, how to install it and
  what has been measured.
- [API contract](docs/api.md): the problem, the callbacks, the solver options
  and the results.
- [Methods and limits](docs/design-notes.md): how the solver works, and
  [what it lacks](docs/gaps.md) compared with mature SQP codes.
- [Application tutorials](tutorials/README.md): seven worked models, each
  checked against an independent reference.
- [AI context](ai-context.md): modeling rules, options and checks for an agent
  that formulates a problem.
- [Shared job API](docs/job-api.md): SQP jobs through a shell or MCP.

Plans, iteration logs, benchmark tables and raw evidence are not rendered
here. Links to them lead to the
[repository on GitHub](https://github.com/dietmarwo/SQP-craft), where the same
Markdown files are the canonical source of every page on this site.

SQP-craft is one of
[18 related repositories](https://github.com/dietmarwo/fcmaes-rust/blob/main/docs/repositories.md).
To choose between it and the other solvers, start at the
[shared solver-selection guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md).
