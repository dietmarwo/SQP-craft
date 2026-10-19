# Job API validation

> This is the delivery's validation. The review before the merge corrected
> three bugs and several robustness problems; see [REVIEW.md](REVIEW.md) for
> those corrections and for what was re-tested on `main`.

Tested on Linux x86_64 with Rust 1.99.1 (2026-10-13). The API layer declares
Rust 1.88 as its minimum; that exact toolchain was not separately tested.
All commands use the delivered lockfiles. Solver cores are unchanged from the
base commits listed in [PLAN.md](PLAN.md) (`SOURCE-REVISION` in the delivery
archive); for SQP-craft that is the 0.16 delivery head `7bc997f`.

| Gate | Observed result |
| --- | --- |
| Shared API/runner and native adapters | Release builds passed in all three independent workspaces |
| Job layer format / Clippy | Passed; Clippy used all targets and `-D warnings` |
| Shared contract/durability tests | 18 passed, including legacy validation and float round-trip regression |
| fcmaes selected Rust tests, release | 30 passed (includes shared contracts, facade and native worker tests) |
| Legacy fcmaes runner/worker tests, debug | 11 passed, including 6 real CLI lifecycle tests |
| Original fcmaes Python integration suite | 14 passed unchanged |
| Mixed integration with all three native workers | 10 passed, none skipped |
| LP-IP local integration | 8 passed; 2 tests for other solvers intentionally skipped |
| SQP local integration | 8 passed; 2 tests for other solvers intentionally skipped |
| LP-IP core/native/adapter Rust selection | 196 passed |
| SQP core/solver/QP/parallel/adapter Rust selection | 197 passed; 1 optional documentation census ignored |
| Tutorial executions | 9 examples completed; numerical/termination assertions passed |
| systemd detached acceptance | Not verified: no systemd user manager; explicit skip (exit 77) |

Counts above describe test invocations and overlap; they are not a claim of
unique test cases. Python extension suites were not re-run for unchanged
bindings. Build targets, toolchains and node executables are not bundled.

## Numerical and lifecycle coverage

The mixed harness uses deterministic retry seeds, independent objective/bound
checks, known LP/MIP optima, infeasible and unbounded LPs, analytic derivative
checks and independent SQP KKT acceptance for both QP backends. It compares
serial and eight-task layouts; LP/SQP also run with one task and eight inner
threads. These are correctness/resource checks, not performance benchmarks.

Lifecycle cases include strict schema rejection, unchanged-key idempotency,
conflicting keys, over-budget resource rejection, model checksum failure,
artifact escape, duplicate journals, queued cancellation under a shared
capacity of eight, forced timeout with unknown active-work totals, killed
launcher with a live worker, conservative slot retention and repair without
solver resume. Native snapshots and checked journals are retained for all
nine tutorial examples in the respective repository archives; this
repository keeps its three under `validation/job-api/tutorial-runs/`.

## Fixes found during acceptance

The first LP assertion incorrectly required bit-exact equality between a
conservative bound and objective 7. It now verifies tolerance agreement and
that the bound remains on the conservative side; no solver values were changed.

The first fcmaes tutorial revealed that the default JSON parser could shift
small f64 values by one bit on a repeated parse, causing manifest validation to
reject a completed journal. The shared crate now enables `float_roundtrip`.
The regression test covers small/large exponents, stable spec digests and the
complete journal-to-manifest-to-runner path. All tutorial examples were rerun.

Extracting the legacy runner also required retaining its original test-only
dependencies. All-target compilation and the unchanged CLI tests verified that
compatibility surface, including the original live-worker repair diagnostic.

## Reproduce

Use the commands in `docs/job-api.md` and the repository's
`tutorials/job-api/README.md`. Each ZIP builds independently. To reproduce the
mixed test, build each repository's worker and pass all three absolute paths to
`scripts/job-api/test_integration.py`. The exact native test selections are:

```sh
# fcmaes-rust
cargo test --release --locked -p optimization-job-api -p optimization-runner \
  -p fcmaes-job-api -p fcmaes-job-worker -p fcmaes-native-worker
cargo test --locked -p fcmaes-runner -p fcmaes-native-worker
python3 scripts/runner/test_integration.py

# LP-IP-craft
cargo test --release --locked -p lp-ip-craft -p lp-ip-craft-native -p lp-ip-job-worker

# SQP-craft
cargo test --release --locked -p sqp-craft -p sqp-craft-solver \
  -p sqp-craft-activeset -p sqp-craft-clarabel -p sqp-craft-parallel -p sqp-job-worker
```

`validation/job-api/` contains raw logs, tutorial results, test environment and
binary hashes in the recorded attempt provenance. Those paths refer to the
validation host and are evidence, not portable execution configuration.
`docs/job-api/SHARED-SOURCES.sha256` verifies identical shared implementation
files across the three archives; in this repository it lists the reviewed
shared files, which differ from the delivered ones (see [REVIEW.md](REVIEW.md)).
The delivery archive's `MANIFEST.sha256` covers every delivered file except
itself and can be checked with `sha256sum -c MANIFEST.sha256` after unzip.

## Synchronization as 0.1.1 (19 October 2026)

The shared files were replaced by the 0.1.1 set that fcmaes-rust, LP-IP-craft
and SQP-craft now share byte for byte (20 files, `SHARED-SOURCES.sha256`).
It keeps this repository's review fixes. Two changes come from the other repositories:
the repair of a queued job whose launcher died now also records zero work with
complete accounting (fcmaes-rust's fix, limited by this repository's rule that
detached attempts with a unit stay unknown), and the result contract knows the
fcmaes set and archive claims. `scripts/job-api/test_review.py` and
`fake_worker.py` are now shared; the reboot test uses whichever worker the
repository builds, and the queued-launcher test also checks the zero-work
accounting. CI also checks the manifest. The SQP worker, the solver crates and
`sqp.solve/1` are unchanged.

On Linux x86_64 with Rust 1.97:

| Gate | Observed result |
| --- | --- |
| `cargo fmt --check`, Clippy `-D warnings` for the job crates | Passed |
| Job crates, release | 21 passed: 20 shared (5 legacy spec, 7 contracts, 8 durability), 1 SQP adapter |
| `scripts/job-api/test_integration.py` with the SQP worker | 9 passed; the fcmaes and LP cases skipped |
| `scripts/job-api/test_review.py` with the SQP worker | 7 passed |
| `tutorials/job-api/run.py` | 3 examples × 8 tasks passed |
| `sha256sum -c docs/job-api/SHARED-SOURCES.sha256` | 20 files OK |
| Mixed node: `test_integration.py` with all three workers, under each repository's runner build | 12 passed, none skipped, three times |


## Result contract 0.2 (19 October 2026)

The shared crates moved to 0.2.0 with the result contract 0.2: five new
claims, evidence (levels, checks, refutations, certificate slot), objective
bounds, worker protocol 0.2 with 0.1 still accepted, coded runner errors,
`already_terminal` on cancelling a finished job, and JSON Schemas for the job
envelope and the result manifest. The shared set grew to 23 files with the
two schemas and `scripts/job-api/outcome_cases.json`. CI installs `jsonschema`
and sets `JOB_API_REQUIRE_SCHEMA=1`, so the schema checks cannot be skipped
there.

An independent review before the merge reported 13 defects; all were fixed
and are covered by tests (see the [plan](PLAN.md#6-result-contract-02-october-2026)).
After the fixes, 1.2 million generated outcomes found no disagreement between
the JSON Schema and `Outcome::check`, and every case of `outcome_cases.json`
behaves the same as a journal record through the runner.

In this repository the SQP worker records a `converged` status that fails the
recomputed KKT acceptance as a refutation instead of failing the job, maps
`locally_infeasible` and `linear_infeasible` to claims with `claimed`
evidence, adds `sqp.evaluate/1`, and reports the workspace version as its
engine version instead of a hard-coded string. Checked with Rust 1.97 on
Linux x86_64:

| Gate | Observed result |
| --- | --- |
| `cargo fmt --check`, Clippy `-D warnings` for the workspace | Passed |
| Workspace tests (`cargo test --workspace --locked`) | 228 passed, including 26 shared (5 legacy spec, 13 contracts, 8 durability) and 4 SQP adapter tests |
| `scripts/job-api/test_integration.py` with the SQP worker | 13 passed, 3 skipped (fcmaes and LP cases) |
| `scripts/job-api/test_review.py` (`JOB_API_REQUIRE_SCHEMA=1`) | 15 passed |
| `tutorials/job-api/run.py` | 4 examples; jobs and results validated against the schemas |
| Mixed node with all three workers, under each repository's runner build | 13 passed, none skipped, three times |
| `sha256sum -c docs/job-api/SHARED-SOURCES.sha256` | 23 files OK |
| Solver crates | Unchanged |
