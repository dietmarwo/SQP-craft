# Direct backend dependencies and notices

Project source is MIT. Dependencies keep their own licenses; this repository
does not relicense them. `Cargo.lock` records the complete resolved graph. The
license texts of the two direct numerical dependencies are included verbatim
for reference.

| Package | License | Text |
| --- | --- | --- |
| clarabel 0.11.1 | Apache-2.0 | [text](licenses/clarabel-0.11.1-LICENSE.txt) |
| nalgebra 0.33.2 | Apache-2.0 | [text](licenses/nalgebra-0.33.2-LICENSE.txt) |

Rayon 1.12.0 (MIT OR Apache-2.0) provides the bounded worker pools of the
`sqp-craft-parallel` crate; no global Rayon pool is configured.

PyO3 0.30.1 provides the Python binding (MIT OR Apache-2.0). Upstream sources
and licenses of all crates are fetched by Cargo during a source build; they are
not vendored. No wheels or native libraries are published (`publish = false`);
anyone distributing a built wheel must include the notices of the statically
linked crates.

## Benchmark harness

The harness in `bench/` uses tools that are neither vendored nor needed by the
solver:
- **S2MPJ** (Gratton & Toint, BSD-3-Clause): the CUTEst problems in Python.
  `python -m bench.s2mpj fetch` clones it at a pinned commit into a cache
  outside the repository.
- **SciPy** (BSD-3-Clause): SLSQP, a comparator, and S2MPJ's sparse matrices.
- **Uno** (`unopy`, MIT): an optional comparator, installed from PyPI.

The committed results in `docs/benchmarks/` contain only the harness's own
records: problem names, sizes, statuses and residuals.

## Shared agent job API

For durable native jobs with an explicit CPU budget, use the
[shared protocol and runner](job-api.md) and the
[solver-specific tutorial](../tutorials/job-api/README.md).
The job profile is a deliberately smaller, versioned surface than the full
in-process solver API. Unknown profile options are rejected. The job crates
come from [optimization-jobs](https://github.com/dietmarwo/optimization-jobs)
through git, at the revision `Cargo.toml` pins; their build also makes
`optimization-mcp`, through which agents submit and follow these jobs
([Agents and MCP](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#agents-and-mcp)).
