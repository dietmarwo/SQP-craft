# Review of the 0.15 and 0.16 delivery

18 October 2026. The delivery (63 commits on `main` at `4d776bb`: thirty
single-thread iterations to 0.15, then thirty eight-thread iterations to 0.16)
was reviewed, tested and corrected before it was merged into `main`. This page
records what the review found and changed. The delivered
[RESULTS.md](RESULTS.md) and [../performance-30/RESULTS.md](../performance-30/RESULTS.md)
stay as delivered, apart from the corrections listed here.

## Corrections

1. **CI failed on every Python job.** `scripts/performance60.py` read
   `/sys/fs/cgroup/cpu.max`, called `os.sched_getaffinity` and imported
   `resource` unconditionally. Its tests failed on GitHub's Linux runners (no
   `cpu.max` at the cgroup root), on macOS (no affinity API) and on Windows (no
   `resource` module). The driver now records these where the platform provides
   them. All CI jobs pass: Rust with Clippy and tests, `cargo check` with Rust
   1.85, and the Python suite on Linux, macOS and Windows with CPython 3.10 and
   3.14. The suites are therefore no longer validated on Linux only; the
   timings still come from the one recorded host.
2. **Batch solves could nest on one thread.** `batch::try_solve_many` ran its
   outer tasks on a Rayon pool. A Rayon worker that waits for another pool keeps
   busy with its own pool's pending jobs. With inner threads above 1, a worker
   whose model or QP kernel waited for an inner pool therefore started another
   outer task. That task ran the remaining starts on top of the suspended solve,
   whose clock kept running, so time limits fired depending on scheduling. This
   happened with budgets 4 × 2, 2 × 4 and 2 × 2. Outer tasks now run on
   dedicated threads (`sqp_craft_parallel::map_with_init`), which block instead.
   A regression test fails on the delivered code and passes now. Batches with
   inner threads 1, which the delivered measurements use, never nested.
3. **Signed zeros.** Three fast paths returned +0.0 where the general code
   returns −0.0:
   - `CsrPattern::multiply` on rows with one or two entries;
   - the singleton projection in the active-set backend;
   - the diagonal inverse.

   They now return the general code's bits, and tests pin the signs.
4. **Documentation:**
   - `Options.threads` in Python still said "Only 1 is supported";
   - `docs/api.md` gave Clarabel "eight native workers", although its engine
     is serial;
   - Rayon was missing from `docs/dependencies.md`;
   - spaces were missing before numbers throughout the prose ("at
     most256", "immutable0.15 baseline1,0.16").
5. **Repository weight.** The raw JSON files above 50 kB are stored
   gzip-compressed: 15.1 MB become 1.1 MB. The report scripts read either form
   and regenerate both `TABLES.md` files byte for byte. On GitHub, the 0.15 and
   0.16 work is one squashed commit. The checkpoint commits and tags stay in the
   delivery archive's source-history bundle.

An independent agent's differential tests against `4d776bb` found the nesting
bug and the signed zeros. They also confirmed the following:
- the general active-set path is bit-identical to `4d776bb` on about 34,000
  random QP sequences, covering factor reuse, inverse promotion, dimension
  changes, indefinite factors, elastic weights, hard rows and warm starts;
- threads 1 and 8 give identical results at n = 512–520;
- batch results are bit-equal to individual solves for budgets 8 × 1, 4 × 2,
  2 × 4 and 1 × 8.

## Independent check on 1,036 CUTEst problems

This check uses the benchmark harness of the CUTEst benchmark (PR #10: `bench/`,
S2MPJ problems, one independent first-order check per run). It compares the
delivered 0.16 build with the 0.14 results of that benchmark, and 0.16 with
eight threads against 0.16 with one. *Identical* means the same status, check
verdict, iterations, function evaluations and derivative evaluations. Budgets
are 500 iterations and 60 s, as in the benchmark.

| comparison | identical | solved | other changes |
| --- | --- | --- | --- |
| Clarabel, 0.14 → 0.16 | 1,034 of 1,036 | 754 → 753 | none; both changes are runs near the 60 s limit |
| active-set + elastic, 0.14 → 0.16 | 1,032 | 813 → 811 | none; all 4 changes are near the time limit |
| active-set, 0.14 → 0.16 | 977 | 799 → 800 (801 with the final build) | 55 numerical, 4 near the time limit |
| active-set, threads 1 → 8 (0.16) | 1,029 | 800 → 799 | none; all 7 changes are near the time limit |
| Clarabel, threads 1 → 8 (0.16) | 1,033 | 753 → 754 | none; all 3 changes are near the time limit |

- **No false claim and no new error.** The one error is HS25NE (a NaN step in
  the active-set QP), as in 0.14.
- **Clarabel and elastic mode are unchanged.** Apart from runs near the time
  limit, the objective agrees to the last bit.
- **The active-set backend changes on unconstrained problems.** This comes
  from the 0.15 direct QP path. A QP without any finite bound is solved in
  closed form by Cholesky, not by the dual method, so its step differs in the
  last bits and the iterates diverge.
  - 50 of the 55 numerical changes are unconstrained problems.
  - Another 78 unconstrained problems and one bound-constrained problem
    keep their counts but end with an objective that differs by up to 2·10⁻⁷
    relative.
  - Five problems are newly solved: BROWNDEN, COOLHANSLS, HIMMELBF,
    PENALTY2 and SSINE. ENGVAL2NE and NONSCOMPNE, whose points already passed
    the check, now also end `converged`.
  - Three are lost: MGH17LS and LUKSAN13LS end in `line_search_failure`, and
    YATP2CLS at the iteration limit. LUKSAN13LS shows only in the final-build
    run, because the first run reached the time limit before the failure.
  - DRCAV3LQ reached the time limit in the first run only.
  - Where both versions solve a problem, 5 end at objectives more than
    10⁻⁶ apart, relative.
  - The delivery's audit (no changed outer iteration, objective difference
    0.0) holds on its own families. It does not extend to unconstrained CUTEst
    problems. The changes are not errors: every accepted point passes the
    independent check.
- **Eight threads change nothing at CUTEst sizes.** The median time ratio of
  the same interleaved runs is 0.99 (active-set) and 0.97 (Clarabel). No
  selected problem reaches a parallel threshold of the QP kernels (512
  variables, or 32,768 Jacobian entries), and the Python callbacks dominate.
- **The final build** after the corrections above gives the same results as
  the delivered build, apart from runs near the time limit: 1,034 of 1,036
  identical runs with Clarabel and 1,032 with the active-set backend. It solves
  754 problems with Clarabel and 801 with the active-set backend; DRCAV3LQ,
  which the first run lost to the time limit, converges again.
- **Timing between 0.14 and 0.16 is not resolved by this check.** In the
  final-build run, the identical solves took 13–15% longer than in the 0.14
  benchmark. However, the Python callbacks, whose code and inputs are the same
  in both, took 13–15% longer too, and they are 98–100% of the time. The
  machine was slower, so the delivered fixture timings are neither confirmed
  nor contradicted here. The solver's own time, which excludes callbacks, grew
  by 5–8%.

The raw records are in `data/cutest-review-delivered.jsonl.gz` (all five
configurations of the delivered build) and `data/cutest-review-final.jsonl.gz`
(Clarabel and the active-set backend with the final build).

## Accepted as designed

- **The direct QP path can accept a point the general path calls
  infeasible**, when the inconsistency is within the acceptance tolerance.
  Take H = I, g = (−5·10⁻⁹, 1) and the equality rows x₀ = 0 and x₀ = 10⁻⁸.
  The direct step violates each row by 5·10⁻⁹, which is below the 10⁻⁷
  tolerance, so it is accepted as optimal; the dual method reports
  `Infeasible`. Both are consistent with the contract, which judges a QP
  solution by the independent check.
- **Error precedence differs between `solve` and the batch.** A custom backend
  that rejects threads above 1 makes `solve` return `Unsupported` before it
  checks the start, while the batch first reports an invalid start as
  `InvalidInput`.

## Not reproduced here

- **The timings.** The review machine has two cores; the recorded host had
  eight.
- **The recorded toolchain,** Rust 1.99.1 with CPython 3.12.14. The review
  used Rust 1.97.0 and CPython 3.13; CI uses current stable Rust and Rust 1.85.
