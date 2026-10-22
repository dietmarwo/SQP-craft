# Publishing SQP-craft

One tag releases every registry artifact with the workspace version:

| Registry | Package | Workflow |
|---|---|---|
| PyPI | `sqp-craft` (abi3 wheels for Linux x86_64, Windows x64, macOS x86_64 and arm64, plus the source distribution) | `python-release.yml` |
| crates.io / docs.rs | `sqp-craft`, `sqp-craft-parallel`, `sqp-craft-activeset`, `sqp-craft-clarabel`, `sqp-craft-solver` | `publish-crates.yml` |

The version is set in `[workspace.package]` of `Cargo.toml`, in
`pyproject.toml` and in `__version__` of `python/sqp_craft/__init__.py`;
`scripts/package_version.py` fails when they disagree. Crates that depend on
a sibling crate name the same version (`version = "X.Y.Z"` next to `path`).
The Python crate and the job worker are not published (`publish = false`).
Publication is irreversible: never reuse a version.

The job worker depends on [optimization-jobs](https://github.com/dietmarwo/optimization-jobs)
through git at a pinned revision. That is allowed because the worker is not
published; the library crates and the Python package depend on crates.io
crates only.

## One-time registry setup

Modeled on fcmaes-rust:

1. Create a GitHub environment named `release` (protect it if releases
   should need approval).
2. PyPI: create a pending Trusted Publisher for project `sqp-craft`: owner
   `dietmarwo`, repository `SQP-craft`, workflow `python-release.yml`, environment
   `release`. The first tag then creates the project.
3. crates.io: the first version must be published with an API token, because
   crates.io accepts a Trusted Publisher only for an existing crate. There are
   two ways; the first needs only a browser:

   - **Token in GitHub (no local toolchain):** on crates.io, Account Settings
     → API Tokens, create a token with the scopes `publish-new` and
     `publish-update`, restricted to the crates `sqp-craft*`, with a short
     expiry. Add it as the secret `CARGO_REGISTRY_TOKEN` of the `release`
     environment (Settings → Environments → `release`). The release tag then
     publishes the crates in dependency order, skipping versions already
     on crates.io, so rerunning the job completes a partial upload.
   - **From a clean checkout of the release commit:**

   ```bash
   cargo login
   cargo publish -p sqp-craft -p sqp-craft-parallel -p sqp-craft-activeset -p sqp-craft-clarabel -p sqp-craft-solver --locked
   ```

   After the first upload, delete the secret and revoke the token. Then
   configure a Trusted Publisher for each crate (owner `dietmarwo`,
   repository `SQP-craft`, workflow `publish-crates.yml`, environment `release`)
   and set the repository variable `CRATES_IO_TRUSTED_PUBLISHING` to `true`.
   Until it is set, `publish-crates.yml` uploads only with the token secret;
   without it, it runs every check and skips the upload.

## Pre-release checks

```bash
python3 scripts/package_version.py
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo publish -p sqp-craft -p sqp-craft-parallel -p sqp-craft-activeset -p sqp-craft-clarabel -p sqp-craft-solver --dry-run --locked
python -m pip install ".[test]" && python -m pytest -q
python scripts/smoke_python_package.py
```

A manual run of `python-release.yml` (Actions, "Run workflow") builds and
smoke-tests every wheel and the source distribution without publishing.

The release commit must have a clean working tree, the intended version in
all three places and in `Cargo.lock`, a dated `CHANGELOG.md` entry, and
passing CI.

## Tag and publish

```bash
version="$(python3 scripts/package_version.py)"
git tag -a "v$version" -m "Release SQP-craft $version"
git push origin "v$version"
```

The tag must equal `v` plus the version; both workflows refuse any other.

## Post-release verification

```bash
python -m venv /tmp/sqp_craft-check && /tmp/sqp_craft-check/bin/python -m pip install sqp-craft
/tmp/sqp_craft-check/bin/python -c "import sqp_craft; print(sqp_craft.__version__)"
cargo new sqp_craft-check && cd sqp_craft-check && cargo add sqp-craft-solver@"$version" && cargo build
```

Check the PyPI page, the crate pages and docs.rs, then create the GitHub
release.

## Discovery after a release

Follow the [ecosystem publication checklist](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/discovery/README.md).
Package descriptions and README links reach registries only with a new
release. Check the rendered package page after publishing and update the
shared catalog when installation availability changes. Repository topics
and About text are GitHub settings, not fields a commit changes.
