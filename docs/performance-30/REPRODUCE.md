# Reproduce the thirty-iteration series

Base: `4d776bbd89f187fe176be805ab91fd64305a8c44` from https://github.com/dietmarwo/SQP-craft. Final source: local 0.15.0. Numerical dependencies are unchanged. Recorded host: AMD EPYC 9V74, Linux x86_64, Rust 1.99.1, CPython 3.12.14. The declared minimum Rust version was not separately executed. The 256-variable NLP cap and one numerical thread per solve remain.

## Build and correctness

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install maturin pytest numpy scipy
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --release --all-features --locked
cargo build --release --locked -p sqp-craft-solver --examples
maturin develop --release --locked
python -m pytest -q
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked
```

On this managed runtime, forcing the PyO3 library test with `--all-targets` exposed a relocated Python linker prefix and a broken unversioned `libpython3.12.so` symlink. The normal extension/Python suite does not need this workaround. An additional all-target/all-feature run uses a private link to the real versioned library and an explicit native-library search path (linker settings, not numerical optimizations):

```bash
python_lib_dir="$(python -c 'import pathlib, sys; print(pathlib.Path(sys.base_prefix) / "lib")')"
mkdir -p /tmp/sqp-python-link
ln -s "$python_lib_dir/libpython3.12.so.1.0" /tmp/sqp-python-link/libpython3.12.so
RUSTFLAGS="${RUSTFLAGS:-} -L native=/tmp/sqp-python-link" \
LD_LIBRARY_PATH="$python_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
cargo test --workspace --release --all-targets --all-features --locked
```

The first 30 checkpoints each ran the full release Rust workspace tests before timing. Milestones 10 and 20 also rebuilt the Python extension, ran its suite, Clippy and independent random NLP checks. Final release validation additionally covers the short-row refinement and updates exact work-counter tests to the new zero-change diagonal-box path, without relaxing numerical assertions.

## Build the common baseline

The archive contains an applicable patch and `history/sqp-perf30.bundle`. The bundle preserves the original base, the harness checkpoint `sqp-perf30-00`, all 30 source tags `sqp-perf30-01` through `sqp-perf30-30`, and the final release tag. It contains source, not prebuilt machine-specific binaries. If starting from the ZIP, clone the bundle to recover Git history first. Keep the included raw data from the ZIP: large stress dumps/logs are deliberately not committed.

```bash
# Run from the unpacked source root.
git clone history/sqp-perf30.bundle /tmp/sqp-history
git -C /tmp/sqp-history worktree add --detach /tmp/sqp-baseline 4d776bbd89f187fe176be805ab91fd64305a8c44
# Use exactly the same new harness on the old numerical source.
cp crates/sqp/examples/performance_kernels.rs /tmp/sqp-baseline/crates/sqp/examples/performance_kernels.rs
CARGO_TARGET_DIR=/tmp/sqp-baseline-target cargo build --release --locked --manifest-path /tmp/sqp-baseline/Cargo.toml -p sqp-craft-solver --examples
```

For paired Python checks build/install 0.14.0 in an isolated environment and preserve its `sqp_craft` package, including `_native*.so`, in a separate directory. Pass that package's **parent** as `--baseline-python`. Every worker is a fresh process: never replace an extension inside a running Python process. Baseline/final use the same frozen model generators and independent evaluator; the audit checks versions, complete case/mode keys, paired counts and original callback hashes.

## Measure serially after builds and stress tests finish

```bash
python scripts/performance30.py docs/performance-30/data/final-common.json baseline=/tmp/sqp-baseline-target/release/examples/performance,/tmp/sqp-baseline-target/release/examples/performance_kernels final=target/release/examples/performance,target/release/examples/performance_kernels --repeats 7
python scripts/compare_performance.py docs/performance-30/data/final-both-backends.json baseline-active=/tmp/sqp-baseline-target/release/examples/performance:active-set final-active=target/release/examples/performance:active-set baseline-clarabel=/tmp/sqp-baseline-target/release/examples/performance:clarabel final-clarabel=target/release/examples/performance:clarabel --repeats 5
python scripts/performance30_holdout.py docs/performance-30/data/holdout-native.json baseline=/tmp/sqp-baseline-target/release/examples/performance final=target/release/examples/performance --repeats 3
```

`performance30.py` and the holdout driver pin each timed child to one allowed CPU; the legacy multi-backend comparator rotates fresh processes without affinity. One warmup is excluded. Primary medians use seven measured repeats, the cold/reuse comparator five, holdouts three. Rust timing excludes native fixture construction and process startup. The kernel/QP harness times fixed work and includes one-time branch-specific setup inside that fixed-work batch. Repeated-QP batches have 30 calls, including misses/promotion; their cache speedups are not fresh-QP or whole-NLP speedups.

All benchmark results must satisfy unchanged convergence, original-unit KKT and known reference-objective assertions. Holdout failures are retained as failures. No compilation, Python installation or stress worker ran concurrently with timing. The host was not exclusive; raw noisy and slower results remain. There is no claim of statistically monotone improvement on every successive wall-time observation.

The held-out sizes were declared before tuning: orbit 30/48, separable 96/192, restoration 48/96. Iteration 16 added previously missing free-row/fixed/dense-normal QP coverage; iteration 29 added coupled free QPs. The preceding numerical checkpoint was rebuilt with the same extended harness each time. The original baseline and final use identical final harness source. The nine-repeat iteration-17 confirmation and five-repeat coupled-free iteration-27 confirmation are retained.

To retime a source tag, check it out in an isolated worktree, copy the final common kernel harness into that tree, build and preserve both example executables. `performance30.py --nlp-only` accepts any number of `LABEL=PERFORMANCE,KERNELS` entries; the 31-point `series.json` was measured together with rotating/reversed order, not reconstructed by multiplying adjacent speedups. `shortrow-confirmation.json` compares the initial release candidate with the refined short-row path. `*-before-shortrow.json` retains the slower candidate.

## Independent validation and exact tables

```bash
python scripts/performance30_validate.py --baseline-python /absolute/parent/of/preserved/0.14/package
python scripts/performance30_audit.py
SEED=2026 COUNT=1000 cargo test --release --locked -p sqp-craft-activeset --test against_clarabel census -- --ignored --nocapture
python scripts/performance30_report.py
# Optional exported chart:
python -m pip install matplotlib
python scripts/performance30_report.py --plot
```

The complete paired validation has 17,372 pairs / 34,744 solver calls: six 300-model seeds under three settings with both backends and both versions; an elastic seed; the four option variants on baseline/Maratos/linear/MPCC families; and independent NumPy checks of development/held-out fixture sizes. These are repeated configurations, not 34,744 unique mathematical problems. Existing nonconverged outcomes stay visible. The harder QP census is a separate outcome tally; baseline and final retain the same four independent-check rejections and one limit.

`TABLES.md`, `data/progress.csv` and the optional `assets/progress.png`/SVG are regenerated from raw measurements. Source hashes, cache validity arguments, the failure/counter ledger and memory/scope limits are documented in RESULTS.md and VALIDITY.md. The delivery archive's manifest verifies its bytes, including the patch and the source-history bundle.

On GitHub, the 0.15 and 0.16 work reached `main` as one squashed commit. The 63 checkpoint commits and their tags (`sqp-perf30-00` to `sqp-perf30-30`, `sqp-parallel-31` to `sqp-parallel-60`, `sqp-parallel-52-cyclic-experiment`) are in the source-history bundle of the delivery archive (`history/sqp-parallel.bundle`), together with the large raw stress dumps and logs. They are not on a branch of this repository; GitHub keeps the commits under pull request #11. Raw files of more than 50 kB are stored gzip-compressed here (`data/*.json.gz`). Fresh runs write plain JSON, and the report scripts read either form.
