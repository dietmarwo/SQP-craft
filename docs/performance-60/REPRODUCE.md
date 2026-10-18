# Reproduce SQP-craft 0.16 and the eight-worker series

Base is tested local 0.15 commit `7ae0520a9da86fd0b49b9e7e486180ead82a88d8`; original upstream is `4d776bbd89f187fe176be805ab91fd64305a8c44`. The delivery archive contains both incremental and cumulative patches, complete source, raw evidence and a source-history bundle.

On GitHub, the 0.15 and 0.16 work reached `main` as one squashed commit. The 63 checkpoint commits and their tags (`sqp-perf30-00` to `sqp-perf30-30`, `sqp-parallel-31` to `sqp-parallel-60`, `sqp-parallel-52-cyclic-experiment`) are in the source-history bundle of the delivery archive (`history/sqp-parallel.bundle`), together with the large raw stress dumps and logs. They are not on a branch of this repository; GitHub keeps the commits under pull request #11. Raw files of more than 50 kB are stored gzip-compressed here (`data/*.json.gz`). Fresh runs write plain JSON, and the report scripts read either form.

## Build and tests

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install maturin pytest numpy scipy
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --release --all-features --locked
cargo test --release --locked -p sqp-craft-solver --example performance_parallel
cargo build --release --locked -p sqp-craft-solver --examples
maturin develop --release --locked
python -m pytest -q
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
```

The executed environment is Linux x86_64, AMD EPYC 9V74, Rust 1.99.1 and CPython 3.12.14. Numerical dependencies/defaults are unchanged; the new execution crate uses locked Rayon 1.12.0 and its dependencies. Declared minimum Rust 1.85 and other operating systems were not separately executed. The ordinary extension is tested through pytest; forcing the PyO3 library test with `--all-targets` on this managed runtime additionally requires a private link to its real versioned libpython because the unversioned link points at a removed temporary directory:

```bash
python_lib_dir="$(python -c 'import pathlib,sys;print(pathlib.Path(sys.base_prefix)/"lib")')"
mkdir -p /tmp/sqp-python-link
ln -s "$python_lib_dir/libpython3.12.so.1.0" /tmp/sqp-python-link/libpython3.12.so
RUSTFLAGS="${RUSTFLAGS:-} -L native=/tmp/sqp-python-link" \
LD_LIBRARY_PATH="$python_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
cargo test --workspace --release --all-targets --all-features --locked
```

This is a linker-path workaround, not a numerical optimization. See the retained successful all-target log.

## Recover baseline and checkpoints

```bash
# From the unpacked source root; preserve the archive's raw data separately.
git clone history/sqp-parallel.bundle /tmp/sqp-history
git -C /tmp/sqp-history worktree add --detach /tmp/sqp015 7ae0520
# Common original-model harness: checkpoint31 changes only harness/docs.
git -C /tmp/sqp-history worktree add --detach /tmp/sqp-baseline-harness sqp-parallel-31
CARGO_TARGET_DIR=/tmp/sqp-baseline-target cargo build --release --locked \
  --manifest-path /tmp/sqp-baseline-harness/Cargo.toml -p sqp-craft-solver --examples
```

Checkpoint 31 has a retrospective exact-source full-suite recheck because its initial log is incomplete: all 173 tests pass; both logs ship. Iterations 32–60 record all regular release-test results before timing. The original 35/49/57 logs end during zero-test rustdoc phases; final-release rustdoc passes separately.

Tags `sqp-parallel-31` through `sqp-parallel-60` preserve every new iteration; `sqp-parallel-52-cyclic-experiment` preserves the rejected scheduler. The previous 30-iteration history is also included. Binaries are rebuilt, not distributed. Recorded hashes identify the measured executables; rebuild hashes may differ with compiler/build paths. Large raw stress dumps/logs ship in the ZIP but are deliberately not committed.

Build/install 0.15 in an isolated environment and preserve its `sqp_craft` directory including the compiled `_native` extension. Pass that directory's **parent** to independent validation. Workers are fresh processes; never replace an extension in a running interpreter.

## Native APIs

Python can request a native maximum without moving its callbacks:

```python
from sqp_craft import Options, SqpOptions, solve
result = solve(spec, model, start, method="sqp",
               options=Options(threads=8),
               method_options=SqpOptions(qp_solver="active-set"))
```

Existing Python model bodies remain serial. Existing small native models use measured scalar fallbacks, including BFGS under the 256-variable NLP cap. The Clarabel engine remains serial. To parallelize expensive native propagation, implement `NlpProblem::set_threads` and partition independent numeric outputs with `sqp_craft_parallel::Execution`; see `crates/sqp/examples/support/propagation.rs`.

Rust `batch::solve_many::<ActiveSetQp, Model, _>(spec, starts, settings, options, 8, factory, cancellation)` creates independent models on their owning worker and retains a backend per worker. `try_solve_many` accepts a fallible factory. Results preserve input order/errors. Use inner `Options::default()` for eight outer workers; the product of outer workers and inner budget must not exceed 8. Factories are Send+Sync, model/backend instances need not be. Complete details are in docs/api.md and VALIDITY.md.

## Measure only after builds and stress tests finish

```bash
python scripts/performance60.py docs/performance-60/data/final-matrix.json \
  final=target/release/examples/performance_parallel --full --batch-jobs 16 --repeats 3
python scripts/performance60.py docs/performance-60/data/common-baseline.json \
  baseline=/tmp/sqp-baseline-target/release/examples/performance_parallel \
  final=target/release/examples/performance_parallel --baseline --full --batch-jobs 16 --repeats 3
python scripts/performance60.py docs/performance-60/data/final-heldout.json \
  final=target/release/examples/performance_parallel --holdout --repeats 3
python scripts/performance60.py docs/performance-60/data/final-qp.json \
  final=target/release/examples/performance_qp --qp-only --repeats 7
python scripts/performance60.py docs/performance-60/data/final-qp-heldout.json \
  final=target/release/examples/performance_qp --qp-only --qp-sizes 384,640,896 --repeats 3
```

The driver pins each child to the first eight allowed CPUs and records quota/affinity, all samples, binary hashes, wall/CPU time, certificates and failures. Timed child processes run serially; each may use up to 8 workers. One fresh-process warmup is excluded. No failure/invalid or nonfinite certificate can become a speedup. Additional targeted propagation/mixed confirmations are retained.

Native `performance_parallel FAMILY SIZE reuse BLOCK BACKEND INNER WORKERS JOBS` owns independent model state on each worker. Batch timing includes model creation, worker startup/teardown and all jobs; each job also records native solver time. Final batches reuse backend state per worker, while the 0.15 manual baseline constructs a fresh backend per job. This is an intentional tested work reduction; mathematical starts/problems remain identical. Fixed-total-work throughput is not one-solve latency. Outer/inner splits with equal maximum budgets can use different actual concurrency because small kernels bypass pools and compatible owners share resources.

`performance_qp N THREADS` reports three fresh-Hessian coupled QPs per sample with retained worker resources. The QP harness explicitly holds a width-matched resource keeper, so cold resource setup is separate and full process/CPU time is also retained; on small fallback cases this keeper is benchmark overhead, not a worker requirement of the lazy QP API. H changes every call, so factor/inverse cache hits cannot hide the arithmetic. 512/1024-variable cases are standalone QPs; the NLP cap stays 256. The propagation fixture uses 32 development segments and 48/96 heldouts, 4096 RK4 steps and finer reference targets, with setup-inclusive and native-solve time recorded separately. It is a separate surrogate, not a change to old orbit callbacks.

## Independent numerical audit

```bash
python scripts/performance60_validate.py --baseline-python /absolute/parent/of/preserved/015/package
python scripts/performance60_audit.py docs/performance-60/data/stress/full \
  docs/performance-60/data/independent-audit.json
SEED=2026 COUNT=1000 THREADS=1 cargo test --release --locked \
  -p sqp-craft-activeset --test against_clarabel census -- --ignored --nocapture
SEED=2026 COUNT=1000 THREADS=8 cargo test --release --locked \
  -p sqp-craft-activeset --test against_clarabel census -- --ignored --nocapture
```

Full audit: 17,372 calls per mode (0.15 baseline 1, current 1, current 8), 52,116 calls total and 34,744 pair comparisons. It uses the same six 300-model seeds, three workspace/restoration settings, both backends, an elastic seed, four option variants over baseline/Maratos/linear/MPCC families and independent NumPy checks of development/heldout sizes. Calls include repeated configurations, not 52,116 unique problems. The wrapper changes only the thread budget and journals every actual solve and independently evaluated solver-owned certificate/objective/point. Frozen source hashes, expected case keys/counts, versions and consistent extension hashes are required. Known nonconvergence is preserved.

`--quick --current-version VERSION --output PATH` runs the smaller 2760-call development audit. The harder census is a separate 1000-case outcome tally at dimensions≤100; larger parallel kernels have separate 512+ identity/stress/reference-QP tests. Known check rejections, errors and limits remain visible.

After all raw files exist, `python scripts/performance60_report.py` regenerates exact tables, and optional `--plot` uses matplotlib to export the scaling figure. RESULTS.md describes actual retained/rejected experiments, measured results and scope. The archive manifest verifies all delivered bytes, including both patches and the source-history bundle.
