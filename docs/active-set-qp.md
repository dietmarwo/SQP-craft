# Dense active-set QP backend

Version 0.14.0, 16 October 2026. Item 1 of the [order of work](gaps.md#order-of-work):
a dense, warm-startable dual active-set QP solver for the step QPs, in the new
crate `crates/activeset` (`sqp-craft-activeset`). It has no dependencies
beyond the core crate; since 0.16 it also uses `sqp-craft-parallel` for the
bounded Rayon execution of its large kernels. Clarabel stays the default
backend.

```python
result = solve(spec, model, x0, method="sqp",
               method_options=SqpOptions(qp_solver="active-set"))
```

In Rust the backend is `NativeSqp<ActiveSetQp>`.

## Result

On the four benchmark fixtures the SQP path is unchanged: the same steps and
the same objectives as with Clarabel. The QP time falls by a factor of 6.0
to 7.7, and the whole solve gets 5.4 to 6.4 times faster:

| Fixture (fixed callbacks) | Clarabel, best mode (ms) | Active set (ms) | Solve speedup | QP time, Clarabel → active set (ms) | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Orbit transfer, 84 variables | 402.9 | 62.7 | 6.4× | 389.5 → 50.7 | 104 / 104 |
| Orbit transfer, 164 variables | 2,985.1 | 545.7 | 5.5× | 2,910.8 → 486.0 | 167 / 167 |
| Separable, 128 variables | 204.9 | 34.0 | 6.0× | 199.6 → 29.6 | 21 / 21 |
| Restoration, 64 variables | 9.0 | 1.7 | 5.4× | 8.6 → 1.25 | 4 / 4 |

These are medians of five rotated rounds after a warmup, with fresh
processes, run by `scripts/compare_performance.py` on a 2-core Intel Xeon VM
at 2.8 GHz. The raw records are in
[performance/active-set-comparison.json](performance/active-set-comparison.json).
"Best mode" is the faster of reused and cold Clarabel workspaces. The
active-set times are for reused workspaces; the reuse setting has no effect
on this backend.

At 164 variables one step QP takes about 2.9 ms: 0.3 ms for the Cholesky
factorization and 2.4 ms for the solve. Clarabel needs about 17 ms for the
same QP (8.6 interior-point iterations). The orbit QPs have 125 equality
constraints (122 defect and boundary rows plus 3 fixed states) and no active
inequalities, so the solve is the factorization of the equality working set.

On the 1,500-problem stress family both backends end with the same status in
all 4,500 paired runs: three settings (cold, reuse, restoration off) times
seeds 1–5 times 300 problems. Both converge in 1,485 of 1,500 runs with
restoration (with cold and with reused workspaces) and in 1,219 without, and
neither ever claims a false convergence.
Of the 4,189 runs that converge with both, 4,136 take the same number of
iterations. The active-set backend takes fewer iterations in 53 and more in
none ([summary](performance/active-set-stress-summary.json)). The 13-case
SLSQP comparison accepts all 13 results, with the same iteration counts as
Clarabel.

## Method

The dual method of Goldfarb and Idnani (1983), as in quadprog, with these
choices:

- **Factorization.** The model's lower triangle is copied into a dense
  buffer and factorized as `H = L Lᵀ` (`dense.rs`, plain loops over
  contiguous columns, skipping zeros, so block-diagonal Hessians cost only
  their blocks). A failed pivot returns `QpOutcome::NotPositiveDefinite`.
  This is the exact positive-definiteness gate that the driver relies on, at
  no extra cost. In the padded regression it fires in four of six runs, and
  the solve recovers by resetting `B`.
- **Factors.** `J = L⁻ᵀ Q` (`n × n`) and the upper-triangular `R` of
  `L⁻¹ N = Q [R; 0]`, where `N` holds the oriented normals of the working
  set. Every bound side is a constraint `σ aᵀp ≥ σ b`: the lower side has
  `σ = 1`, the upper side `σ = −1`. An equality (equal bounds) is never
  dropped, and its multiplier has either sign. Normals are read from the CSR
  rows. A variable bound is a unit vector.
- **Adding a constraint** reflects the part of `d = Jᵀn` outside the working
  set into one entry with a Householder reflector on those columns of `J`
  (`4 n (n − q)` operations). A chain of Givens rotations costs `6 n (n − q)`;
  on the 164-variable orbit the reflector cut the QP engine time by about 15%
  (single runs).
  **Dropping** a constraint restores `R` with Givens rotations.
- **Iterates** stay dual feasible: `x` minimizes the objective on the working
  set, and the inequality multipliers are nonnegative. The most violated
  constraint (violation divided by the normal's norm) is added. Blocking
  multipliers that reach zero are dropped on the way (a partial step). A
  constraint that depends on the working set and has nothing to drop proves
  infeasibility. No feasible start is needed.
- **Initial working set.** All equalities, plus the warm-start constraints
  (`QpWarmStart.active_set`, `AtLower` or `AtUpper` on finite sides). A
  constraint that depends on the ones before it is skipped. Then `x` and the
  multipliers are computed on that set, and the most negative inequality
  multiplier is dropped until none is negative. Any working set is therefore
  a valid start: a wrong guess costs drops, not correctness.
- **Polishing.** At the start, at the end, and before any infeasibility
  claim, `x` and `u` are recomputed from the factors
  (`x = J₁R⁻ᵀb − J₂J₂ᵀg`, `u = R⁻¹(R⁻ᵀb + J₁ᵀg)`). One step of iterative
  refinement against the original `H` and normals follows. An infeasibility
  claim is made only from a polished point, so drift in `x` from many updates
  cannot cause it.
- **Dependence** is tested absolutely, at the rounding level of `d`:
  `‖d₂‖ ≤ 100 ε √n ‖J‖_F ‖n‖`. `‖J‖_F` is computed once, since reflections
  and rotations keep it. A first version used a relative test
  (`‖d₂‖/‖d‖` against `ε κ(L)`). On a nearly singular `B` (seed 4, case 86 of
  the stress family, `det B ≈ 10⁻⁹`), `L⁻¹` maps independent normals to
  nearly parallel vectors, so that test called a feasible QP infeasible. It is
  now a unit test.
- **Partial steps** ignore multiplier-direction entries below the rounding
  level, `10³ ε √n κ̂ · max |R⁻¹d₁|` with `κ̂ = max Lᵢᵢ / min Lᵢᵢ`. Otherwise a
  dependent constraint could take a dual step of 10¹⁶.
- **Tolerances.** A constraint is violated when it misses its bound by more
  than `qp_tolerance · (1 + |b| + Σ|aᵢxᵢ|)`. A multiplier counts as negative
  below `−qp_tolerance · (1 + max |u|)`.

## Contract details

- **Outcomes:** `Optimal` (after `QpModel::accepts` with the driver's
  acceptance tolerance), `Infeasible`, `NotPositiveDefinite`,
  `LimitReached(None)` with `limit_reason` `"iterations"` or `"time"`, and
  `Cancelled`. `Unbounded` cannot occur, since `H` is positive definite. An
  independent-check failure on a cold start is `Error::Backend`, which the
  driver reports as `qp_failure`.
- **Cold confirmation.** A warm-started attempt that ends infeasible or
  fails the check is repeated once from a cold start, within the same budget
  (`cold_retries`, `retry_reason`). As with Clarabel's reused workspaces, an
  unconfirmed infeasibility finding never starts restoration. The 31,091 QPs
  of the stress family (seeds 1–5, restoration on and off) needed no retry.
- **Budget.** `QpReport.iterations` (and `qp_iterations` in the run report)
  counts the working-set changes after the initial working set.
  `qp_max_iterations` caps only drops and re-entries: a constraint's first
  entry in an attempt is free. A cold start needs one entry per active
  inequality, which exceeds any fixed cap for large `n + m`: with every
  change charged, the default cap of 200 failed cold random QPs of about 60
  variables and 60 rows. Repeated changes are what degeneracy costs. The cancellation token and the deadline
  are polled at every change.
- **Report:** `conversion_seconds` (the dense copy of `H` and the row norms),
  `setup_seconds` (Cholesky, `setups = 1`), `solve_seconds` (`L⁻ᵀ` and the
  active-set iterations) and `check_seconds`. `updates` is always 0.
- **Ignored inputs:** `QpWarmStart.step` and `.multipliers`, which a dual
  method cannot use, and `QpOptions.reuse_workspace`. The three `n × n`
  buffers are always reused, and a result never depends on earlier calls (a
  unit test solves models alternately and compares bit for bit).
- **Size:** the backend itself has no variable cap. Memory is `3 n²` values,
  and each QP costs `O(n³)` for `L` and `L⁻ᵀ` plus `O(n (n − q))` per change.
  The driver's 256-variable limit applies.

## Warm start in the driver

`SqpOptions.qp_warm_start` (default `True`) passes the working set of the last
step QP to the next one, but only when the backend reported one. Clarabel
reports none, so its runs are unaffected: a test compares runs with the flag
on and off bit for bit. On the orbit fixture no inequality ever becomes
active, so the warm start saves nothing there. On a 6-variable problem whose
bounds are all active, a cold QP makes 6 changes and a warm one makes 0 after
the first QP (`tests/test_active_set.py`, `crates/sqp/tests/active_set_backend.rs`).

## Validation

- Unit tests (`crates/activeset/src/tests.rs`, `dense.rs`):
  - closed-form solutions and the multiplier signs of lower, upper and
    equality sides;
  - inconsistent rows, equalities and boxes;
  - duplicate, scaled, zero and dependent rows;
  - indefinite, singular and NaN Hessians;
  - warm starts from the exact set and from a mostly wrong set;
  - the budget rule, cancellation, the time limit and invalid input;
  - block-diagonal Hessians;
  - the near-singular regression;
  - the dense kernels (`Jᵀ H J = I`, the reflector keeping `Jᵀn`).
- Random QPs against Clarabel (`crates/activeset/tests/against_clarabel.rs`):
  - 1,000 feasible QPs with up to 40 variables, regularization down to 10⁻⁶,
    and duplicate rows: wherever Clarabel succeeds (at least 950), the same
    objective (to 10⁻⁶ relative, never worse than Clarabel's by more than
    10⁻⁷) and the same step (to 10⁻⁴ relative);
  - 150 infeasible QPs, which both backends find infeasible;
  - 600 warm starts: the QP's own working set needs no change, a perturbed
    QP gives the cold solution, and warm starts need less than half the
    changes.
- An ignored census test runs harder QPs: up to 100 variables, regularization
  down to 10⁻⁸ and every fourth QP made infeasible. On 4,000 such QPs
  (seeds 1 and 2), 3,025 feasible and 975 infeasible:
  - The active-set backend fails 14. Thirteen fail the independent check
    (primal residual up to 5·10⁻⁷, or stationarity up to 4·10⁻⁶) and one
    reaches the change cap. All 14 are QPs with regularization 10⁻⁸ and at
    least 51 variables, where `κ(H)` is around 10¹⁰ or more.
  - Clarabel also fails 13 of those 14. It fails 40 feasible QPs that the
    active-set backend solves, and 14 infeasible ones that the active-set
    backend finds infeasible.
- Driver and Python:
  - the padded Hessian-gate regression with both backends;
  - the orbit transfer (same objective to 10⁻⁹, same steps);
  - HS071 with restoration;
  - the warm-start counts;
  - the option validation (`ValueError` for an unknown `qp_solver`).

## Limits and next steps

- The method is dense. The `O(n³)` work per QP (Cholesky and `L⁻ᵀ`, about a
  fifth of the orbit QP time) is repeated every iteration, although `B`
  changes by rank two.
- On QPs with `κ(H)` around 10¹⁰ the accuracy can fall short of the
  acceptance rule. Clarabel fails on these QPs as well.
- Clarabel stays the default until there is more evidence, for example on
  the target application's models. Switching the Python default changes one
  line in `SqpOptions` and one in the binding; Rust callers choose the
  backend type themselves.
- The next item in the order of work, elastic mode (gap 4), can be built on
  this backend: an ℓ1 penalty on the constraints inside the QP keeps it
  feasible and uses the same factors.
