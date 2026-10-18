# Review of the job API 0.1.0 delivery

18 October 2026. The delivery, `SQP-craft-job-api-0.1.0`, was reviewed, tested
and corrected before it was merged into `main`.
- **What it contains:** `PATCH-job-api.diff` on `7bc997f`, the unreviewed 0.16
  head. The patch adds the shared job crates, the SQP worker, the guide, the
  tutorial and a CI workflow.
- **Port to `main`:** the patch applied to `main` (the reviewed 0.16 and the
  CUTEst harness) without change, apart from `docs/dependencies.md`, where
  both new sections are kept. The patch changes no numerical crate, so the
  worker here builds on the reviewed 0.16 rather than on the delivered one.
- **Evidence:** the delivery's validation logs and tutorial attempts (216 kB)
  are now under `validation/job-api/`, where [VALIDATION.md](VALIDATION.md)
  points.

This page records what the review found and changed. The delivered
[PLAN.md](PLAN.md) stays as delivered. [VALIDATION.md](VALIDATION.md) and the
[guide](../job-api.md) are corrected as listed here.

## Corrections

1. **CI failed with Rust 1.85.** The job crates declare `rust-version = "1.88"`
   (the runner uses a let chain), so `cargo check --workspace` with the
   project's Rust 1.85 failed.
   - The 1.85 job now excludes the three job crates, and a new job checks them
     with Rust 1.88.
   - `job-api.yml` ran twice per pull-request commit (`on: [push,
     pull_request]`). It now has the same triggers, permissions, concurrency
     and cache as `ci.yml`.
2. **An unreadable journal left the job `running` and broke `jobs list`.**
   - If a worker wrote an invalid record to `results.jsonl`, the launcher could
     not build the result manifest and dropped the error. It never wrote
     `run.json` or a terminal state.
   - `jobs get`, `jobs list` (for every job on the node) and `jobs repair` then
     all failed with "expected ident at line 1 column 2".
   - Such an attempt is now sealed `failed`, with no results, `work` null and
     `runner.journal_error`. Repair seals a corrupt journal the same way instead
     of refusing forever.
   - Replay errors name the journal. `jobs list` reports the error for that job
     only. `jobs get` still reports it as an error.
3. **A launcher killed while its attempt was queued could never be repaired.**
   - Without `worker-process.json`, only terminal states counted as "processes
     gone". `jobs wait` therefore said "run `jobs repair`", and repair then
     refused: "liveness unknown".
   - The attempt reaches `running`, durably, before the worker is spawned. So a
     `waiting` attempt, or a foreground one not yet launched, whose locks are
     free never had a worker. Repair now seals it `interrupted`.
   - Detached `submitted` and `starting` attempts may still be started by their
     unit, so they stay unknown.
4. **After any reboot, every finished job was reported
   `interrupted_after_reboot`.** Only non-terminal states are flagged now.
5. **Validation needed an executable `TMPDIR`.** The runner staged the worker
   copy under `TMPDIR`, so on a host with `/tmp` mounted `noexec` every
   `validate` and `run` failed with "Permission denied". It now stages under
   the node root (`.staging-*`). The root must allow execution anyway, because
   attempts run their staged worker there.
6. **A signal during validation left the validator running.** The validator
   runs in its own process group, which a terminal's SIGINT does not reach.
   The runner exited, the validator kept running, and the staging directory
   with its copy of the worker stayed behind. The runner now handles the
   signal, and the new `process::bounded_command_until` stops the group.
7. **An append after a torn journal tail destroyed every earlier record.**
   - An append that is cut short (a full disk, an I/O error) leaves a final
     fragment without its newline. The next append joined the fragment and its
     own record into one complete but invalid line. Every earlier record became
     unreadable, and `repair_tail` refused.
   - The worker SDK could do this: after one task's append failed, another
     task still appended its result.
   - Appends now end a torn tail first, the way readers already treat it: a
     valid record gets its newline, and anything else is dropped. A test
     appends after both kinds of tail.
8. **Two directory entries were not synced:** the new job directory, which
   `requests.json` then points to, and `results.jsonl`.
9. **Smaller changes:**
   - `./model.json` is accepted as an input path; `.` and `./` still name no
     file.
   - Detached units for common jobs are named `optimization-job-ID.service`;
     legacy fcmaes jobs keep `fcmaes-ID.service`.
10. **Documentation:**
    - The 4 MiB validation-output cap also bounds the specification, because
      validation prints it back. About 600,000 one-dimensional SQP starts fit,
      not the 1,000,000 tasks the envelope allows.
    - With a key that already names a job, `run` returns that job's current
      state, which need not be final.
    - The tutorial says that for the registered models `inner_threads` above 1
      changes nothing. The models have at most 256 variables and one row, and
      the QP kernels split their work only from 512 variables or 32,768
      bounded-row entries.
    - [VALIDATION.md](VALIDATION.md) no longer points to `SOURCE-REVISION` or
      `MANIFEST.sha256` as if they were in the repository.

Corrections 2–8 have regression tests:
- `scripts/job-api/test_review.py` runs the release runner against a protocol
  fixture (`fake_worker.py`) and the SQP worker. Six of its seven tests fail on
  the delivered runner; the seventh checks that slots are released after a
  crash. It runs in CI.
- The job-api crate's tests cover the torn-tail append and the `./` paths.

## Shared sources

The guide, the job crates and the integration harness are meant to be
identical in fcmaes-rust, LP-IP-craft and SQP-craft. Corrections 2–9 change the
following shared files:
- `crates/optimization-job-api/src/{process,result,spec,storage,worker}.rs`;
- `crates/optimization-job-api/tests/{contracts,durability}.rs`;
- `crates/optimization-runner/src/lib.rs`;
- `docs/job-api.md`.

[SHARED-SOURCES.sha256](SHARED-SOURCES.sha256) now lists the reviewed hashes.
The other two repositories need the same files when their job-API deliveries
are merged. `scripts/job-api/test_review.py` and `fake_worker.py` work for any
worker and can go with them.

## Checked and left as delivered

An independent agent tested the delivery against its documentation and found
corrections 2–7. It also confirmed the following:
- **Admission:**
  - Six concurrent runs on a capacity of 4 never held more than 4 slots.
  - Slots were released after a worker crash (SIGSEGV), a timeout,
    `cancel --now` (under 2.5 s) and SIGINT.
  - Slot counts of 0 and 1025, and a product that overflows, are rejected.
- **Specification limits:**
  - Rejected: wall times of 0, below 0 or above one year; grace below 0 or
    above its limit; `1e400`; request keys of 0 or 201 bytes; a wrong
    `task_count`; unknown fields at every level; `inner_threads` = 9.
  - Accepted: a 200-byte multibyte key.
- **Input staging rejects:**
  - file symlinks and symlinked parent directories that leave the spec
    directory;
  - `..` and absolute paths;
  - wrong or uppercase hashes;
  - files above 64 MiB, a 17th input, and a directory given as a file.

  The bytes that are hashed are the bytes that are copied.
- **Journals and recovery:**
  - Snapshots are written to a temporary file, synced, renamed, and the
    directory synced.
  - A torn tail after a crash gives `failed` with `work` null.
  - After launcher loss, a live worker keeps its slots, and repair keeps the
    worker's exact result.
  - The same key with the same spec gives one attempt; a changed spec is
    rejected.
- **SQP worker:**
  - `first_order` is claimed only after the worker's own KKT recheck passes.
  - Both backends reach the known optima at n = 3–5, including the 8/1/8
    layout: Rosenbrock (1, …, 1), simplex 1/n and the shifted quadratic
    (i+1)/n.
  - Deadlines and SIGINT stop cooperatively with complete accounting.
  - Results are ordered by `task_id`, and validation calls are counted
    separately.
- **Logs and detached runs:** logs are capped at 1 MiB, and `submit` without a
  systemd user manager refuses cleanly.

Left as delivered:
- Duplicate JSON keys are accepted, and the last one wins, because specs are
  parsed through `serde_json::Value`.
- A worker descendant that escapes into its own session holds the launcher until
  it exits. The guide places escaped sessions outside the foreground contract.
- `jobs get` on an attempt with a corrupt journal still fails; the job's state
  and `run.json` say `failed`.
- The worker reports engine version `0.16.0` as a literal.

## Not tested

Detached execution under systemd (no user manager here) was not tested. Neither
were the LP-IP and fcmaes workers (including the `+infinity` gap encoding), the
legacy fcmaes path, power loss, or a full disk end to end. The review machine
has Rust 1.97; Rust 1.88 runs in CI only.

## Verification

- **Job crates:** 20 Rust tests pass (19 delivered and one new). Clippy with
  warnings denied and `rustfmt` are clean.
- **Python harnesses:**
  - The integration harness passes 8 tests; 2 are skipped because the LP-IP
    and fcmaes workers are not built here.
  - The review regressions pass 7 tests.
  - The tutorial runs its three examples.
- **CI:** all jobs pass, including Rust 1.85 for the solver crates and Rust
  1.88 for the job layer.
