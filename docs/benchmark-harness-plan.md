# Plan: a benchmark harness for decisions

Version 0.14.0, 17 October 2026. This is item 1 of the survey's
recommendations (project document `claude/sqp-literature-survey.md`) and item 6
of the [order of work](gaps.md#order-of-work), gap 6 ("thin external
evidence").

## Why

The solver now has several opt-in mechanisms whose defaults are open:
- **The QP backend:** Clarabel or active set.
- **The merit:** ℓ1, ℓ1 with second-order correction, or augmented
  Lagrangian.
- **The BFGS repair.**
- **Scaling.**
- **Declared linear rows.**
- **Elastic mode or restoration.**

The test families in the repository were written to show each mechanism.
They cannot say which default is right on problems nobody tuned for. The
harness supplies that evidence on a public collection, judged by one
independent check, with external comparators, in a form that can be rerun.

## What it answers

| Question | Comparison |
| --- | --- |
| Q1. Default QP backend | `clarabel` vs `active-set`, both with the ℓ1 merit |
| Q2. Default merit | `active-set` with ℓ1, with ℓ1 + second-order correction, and with the augmented Lagrangian; the augmented Lagrangian also with Clarabel |
| Q3. BFGS repair | `active-set` with `hessian_update="augmented"` |
| Q4. Scaling | `active-set` with `scaling="jacobian"`, and with the linear rows declared and `scaling="linear"` |
| Q5. Linear rows | `active-set` with the S2MPJ linear rows (`lincons`) declared |
| Q6. Elastic mode | `active-set` with `elastic_mode=True` |
| Q7. Where sqp-craft stands | SciPy SLSQP, Uno with the filterSQP preset (L-BFGS, trust region, filter, BQPD), Uno with the IPOPT preset (L-BFGS, interior point, MUMPS) |
| Q8. What second derivatives would buy (gap 2) | Uno filterSQP preset with exact Hessians from S2MPJ |
| Q9. What fails, and why | Failure classes of the best sqp-craft configuration, compared with the problems a comparator solves |

## Problems

**Source.** S2MPJ (Gratton & Toint): CUTEst problems decoded to pure Python,
BSD-3-Clause, pinned to commit `fea6a70` (24 August 2026). It is cloned into
a cache outside the repository (`python -m bench.s2mpj fetch`) and not
vendored. Every problem uses its default parameters.

**Catalog** (`python -m bench.catalog`). Each of the 1,104 problems is set up
in its own subprocess, with a 60 s limit. The catalog records:
- the class string, `n`, `m` and the number of linear rows;
- the setup time;
- whether the start is defined, meaning that f, c, ∇f and J are finite
  there;
- the time of one evaluation.

The catalog is committed, so the selection is reproducible without rerunning
it.

**Selection** (`bench/selection.py`):
- `n ≤ 256`, the dense cap, and `m ≤ 512`;
- smooth class (`R` in the class string);
- defined at the start;
- setup within 30 s, and one evaluation of f, c, ∇f and J within 0.2 s, to
  bound the run time.

The timing rules vary between catalog runs, so the selection is frozen once
made (`--frozen`); later catalog runs update only the problems' attributes.

**Subsets reported separately:**
- unconstrained;
- bound-constrained (including fixed variables);
- linearly constrained;
- nonlinearly constrained;
- a control-like subset: DTOC*, CATMIX, LAUNCH, ROBOT, HANGING, CATENARY,
  AIRCRFT*, and others the catalog shows to be discretized control problems.

**Start points.** S2MPJ's `x0`, clipped into the variable bounds, which
sqp-craft requires (the HS problems often start outside them). Every solver
gets the same clipped start.

## The adapter (`bench/adapter.py`)

One `S2mpjProblem` serves every solver.
- **Bounds** with `|b| ≥ 10²⁰` are infinite.
- **The Jacobian pattern** comes from the problem structure: the linear rows
  of `A` plus the variables of each group's elements. It is verified against
  evaluated Jacobians at the start and two nearby points, and extended where
  needed; the catalog records when that happens.
- **Linear rows** are S2MPJ's `lincons`, less the rows whose group has a
  nonlinear group function or whose Jacobian changes between the sampled
  points: S2MPJ ignores group functions when it marks rows linear.
- **Evaluations** are counted the same way for every solver, as distinct
  points. One *function evaluation* is a point where f and c are computed,
  one *derivative evaluation* a point where ∇f and J are computed. Solvers
  that ask for f and c, or ∇f and J, separately at the same point are not
  charged twice.

## Solvers

**Budgets for all:** 500 iterations, 60 s per solve, and the same start.
- **sqp-craft variants:**
  - `max_callback_calls = 5000`, otherwise defaults: absolute tolerances
    10⁻⁶ and feasibility 10⁻⁷;
  - the variants of Q1–Q6, plus one combination of all options from SNOPT;
  - the options are the documented ones, and nothing is tuned per problem.
- **SciPy SLSQP:** `ftol = 1e-12` and `maxiter = 500`, as in
  `compare_baseline.py`. Equality and inequality rows go in one vector
  constraint each.
- **Uno:**
  - presets `filtersqp` (L-BFGS Hessian, BQPD) and `ipopt` (L-BFGS, MUMPS);
  - `max_iterations = 500`, `time_limit = 60`, logger silent;
  - the exact-Hessian run uses S2MPJ's constraint and objective Hessians,
    dense lower triangle, sign convention `MULTIPLIER_NEGATIVE`.

**Isolation.** Each problem runs in a worker subprocess that solves it with
every solver in turn and reports one JSON line per solve. A solve that
exceeds 90 s, or crashes the worker, is recorded as `timeout` or `crash`,
and the worker restarts with the remaining solvers. Two workers run in
parallel.

## The independent check (`bench/check.py`)

Every result is judged by the same first-order check, whatever its solver
claims. It is computed in NumPy from S2MPJ's own values at the returned
point:
- primal violation of rows and bounds ≤ 10⁻⁶;
- stationarity, complementarity and dual-sign residuals ≤
  10⁻⁶·max(1, ‖∇f(x)‖∞).

The thresholds are relative to the gradient, so solvers with relative
internal tests are not penalized on badly scaled CUTEst problems. They are no
stricter than sqp-craft's own absolute 10⁻⁶ test.

**Multipliers.** The check uses the solver's multipliers when they pass.
Otherwise it uses least-squares multipliers on the active set, as in
`examples/compare_baseline.py`, so a sign convention cannot fail a correct
point.

Each run is then classified:
- **solved:** the check passes;
- **solved to the best objective:** solved, and the objective is within
  10⁻⁶·max(1, |f*|) of the best objective of any solved run of that
  problem;
- **false claim:** the solver reports success, but the check fails. For
  sqp-craft any false claim fails the harness.

## Measures and reports (`bench/report.py`)

- **Counts.** Solved and solved-to-best, per solver, per subset and per
  constraint class.
- **Costs.** Function evaluations, derivative evaluations, callbacks (their
  sum), iterations and wall time.
- **Performance profiles** (Dolan & Moré 2002) over callbacks and over time,
  at τ = 1, 2, 4, 8, 16, 32 and ∞. Failures count as infinite cost.
- **Pairwise comparisons** for the decisions. Profiles of more than two
  solvers cannot rank the others (Gould & Scott 2016), so each decision
  compares the current default with one alternative:
  - problems solved by only one of the two;
  - among the problems both solve, wins and losses in callbacks;
  - the shifted geometric mean of callbacks (shift 10) and of time (shift
    1 s).
- **Failure classes** of the sqp-craft runs: status, by subset, and which of
  those problems a comparator solves.
- **Files:**
  - the raw results as one JSON line per run (`docs/benchmarks/results.jsonl.gz`);
  - the selection, the environment (versions, CPU, the S2MPJ commit and the
    sqp-craft commit) and the report as Markdown
    (`docs/benchmark-results.md`).

## Validation of the harness

- **Unit tests** (`tests/test_bench.py`, no network):
  - the adapter on hand-built S2MPJ-like problems: bounds, start clipping,
    the structural pattern and its extension, linear rows and counting;
  - the check on known KKT points, including a wrong-sign multiplier rescued
    by least squares and a non-KKT point;
  - the profile and geometric-mean arithmetic on synthetic records.
- **A CI job:**
  - it clones S2MPJ at the pinned commit and runs the default and the
    active-set configurations and SLSQP on 20 Hock–Schittkowski problems;
  - it fails on a sqp-craft false claim;
  - Uno is optional there.
- **Spot checks:** the HS problems' objectives against the published optima
  where known, for example HS71 at 17.0140173.

## Deliverables and decisions

- `bench/`: the package `s2mpj`, `catalog`, `selection`, `adapter`, `solvers`,
  `check`, `worker`, `run` and `report`.
- `docs/benchmarks/`: the catalog, the selection and the raw results.
- `docs/benchmark-results.md`:
  - the answers to Q1–Q9 with the numbers behind them;
  - a recommendation for each open default, marked as recommendation;
  - the failure analysis that orders the next work items. The defaults
    themselves are your decision.
- `gaps.md` updated (gap 6 and the order of work); the README and the
  project documents updated.
- A PR with the harness and the results.

## Limits, stated up front

- **S2MPJ is slower than Fortran CUTEst.** Wall times measure Python
  callbacks plus solver work for every solver alike. They are comparable
  between solvers in this harness, not with published Fortran timings.
- **Default problem sizes only,** and only those with `n ≤ 256`. Large
  CUTEst problems are outside the dense pilot's scope.
- **No Ipopt itself:** it needs a compiled library that cannot be installed
  here. Uno's `ipopt` preset reproduces its method.
- **No COPS or Betts problems** beyond the control problems CUTEst contains.
  The orbit transfer stays in the repository's own families.
- **One start per problem**, as in CUTEst practice.
