# Plan: four lessons from SNOPT

Version 0.14.0, 17 October 2026 (results: [snopt-lessons.md](snopt-lessons.md)). The four steps follow the comparison with SNOPT in the
literature survey (project document `claude/sqp-literature-survey.md`). The
reference throughout is P. E. Gill, W. Murray, M. A. Saunders, *SNOPT: An SQP
algorithm for large-scale constrained optimization*, SIAM Review 47 (2005)
99–131 ([PDF](https://ccom.ucsd.edu/~optimizers/static/pdfs/siam_44609.pdf)),
cited below as GMS with section and equation numbers.

## Ground rules

- **Opt-in.** Each step is a new option or problem attribute, and its default
  keeps today's behavior. With every new option at its default, the 9,000
  stress runs of `scripts/validate_performance_stress.py` (seeds 1–5, both QP
  backends) must be bit-identical to `main` (2288a8e).
- **The verification contract stays.** Convergence is still tested on the
  original, unscaled problem with the absolute tolerances and the rounding
  guard. Every convergence claim in every measurement run must pass the
  independent evaluator of `examples/compare_baseline.py`.
- **Measured, not argued.** Every step is measured on the existing families
  and reported as measured, whether it helps or not:
  - stress: 1,500 problems, seeds 1–5;
  - complementarity (MPCC): 600 problems;
  - Maratos example: 200 starts;
  - orbit transfer: the 13 SLSQP-comparison cases.

  Steps 3 and 4 add families that exercise their mechanism. Defaults change
  only if you decide so.
- **One PR.** The version stays 0.14.0, as for the active-set, elastic-mode
  and second-order-correction PRs. One document describes all four steps and
  their results, with updates to `gaps.md`, `api.md` and the README.

## Step 1: Curvature repair instead of Powell damping

**SNOPT** (GMS §2.10). An update is made only if `sᵀy ≥ σ`. Otherwise:

1. **First modification.** Redefine `s` and `y` from `x̄`, the first feasible
   iterate of the QP solve.
2. **Second modification.** Replace `y` by the gradient difference of the
   augmented modified Lagrangian (eqs. 2.7–2.8):

   ```text
   Δy = (J(x⁺) − J(x))ᵀ Ω d_L,   d_L = c(x⁺) − c(x) − J(x)s
   ```

   `d_L` is the departure from linearity. `Ω = diag(ω)` is the minimum-norm
   solution of `aᵀω = β, ω ≥ 0`, with `aᵢ = vᵢ·d_L,ᵢ`,
   `v = (J(x⁺) − J(x))s` and `β = σ − sᵀy`, so `ω = β·a⁺/‖a⁺‖²`.
3. **Skip.** If no solution exists, or `ω` is very large, no update is made.

**Correction to my earlier summary.** In the reply before this plan I
described the added term as `½Σωᵢcᵢ²`, which adds about `JᵀJs`. That was
wrong. SNOPT uses the departure from linearity `d_L`, so the added curvature
comes from the constraint curvature along `s`, not from the constraint
normals. For linear constraints `d_L = 0`, and no repair is possible.

**Here:**
- Option `SqpOptions.hessian_update`:
  - `"damped"` (default) keeps Powell's damping;
  - `"augmented"` applies modification 2, then skip.
- Modification 1 is not transferable. The dual Goldfarb–Idnani method has no
  primal-feasible intermediate iterate, and Clarabel's interior points are
  not vertices.
- The trigger stays at today's `sᵀy < 0.2·sᵀBs`, so the two variants differ
  only in what happens below it.
- "ω very large" has no number in the paper. The update is skipped when
  `‖Δy‖ > 100·‖Bs‖`; the skip count is reported, so this choice can be
  revisited.
- A skipped update leaves `B` unchanged (no reset).
- In block mode, each block applies the same rule to its own part of `s`.

**Measured first.** New counters, also in the default mode:
- `RunReport.bfgs_damped_updates`, `bfgs_modified_updates` and
  `bfgs_skipped_updates`;
- `IterationRecord.curvature` for each iteration.

The first measurement is how often damping fires, and how near the solution.
That decides whether the repair can matter.

**Validation:**
- unit tests of the ω formula against closed forms;
- `sᵀy⁺ = σ` after the repair;
- skips for linear rows;
- a driver test where damping fires near the solution and the repair keeps
  the true curvature;
- the families in both modes.

## Step 2: Augmented-Lagrangian merit and violation limit

**SNOPT** (GMS §2.7–2.8):
- **Merit function** (eq. 2.3):
  `M_ρ(x, π, s) = f − πᵀ(c − s) + ½Σρᵢ(cᵢ − sᵢ)²`.
- **Line search.** It searches along `(x, π, s) + α(p, π̄ − π, ŝ − s)` toward
  the QP solution, with multipliers `π̄` and linearized slacks `ŝ`.
- **Slacks.** `s` minimizes `M` for the current `(x, π)`.
- **Penalty.** `ρ*` solves `min ½‖ρ‖²` subject to
  `φ′(0) = −½pᵀBp, ρ ≥ 0` (LSPρ).
- **Damped update** (eq. 2.4):

  ```text
  ρ̄ᵢ = max(ρ*ᵢ, ρ̂ᵢ)
  ρ̂ᵢ = ρᵢ                    if ρᵢ < 4(ρ*ᵢ + Δρ)
  ρ̂ᵢ = √(ρᵢ(ρ*ᵢ + Δρ))      otherwise
  ```

  Δρ starts at 1 and doubles whenever the trend of `‖ρ‖` reverses.
- **Starting values.** `ρ` and `π` start at 0. After the step,
  `π ← π + α(π̄ − π)`.
- **Violation limit** (eq. 2.5): trial points must satisfy `c(x + αp) ≥ −b`,
  with `bᵢ = τ_V·max(1, −cᵢ(x₀))` and `τ_V = 10`.

**Here:**
- **Option `SqpOptions.merit`:** `"l1"` (default) or `"augmented-lagrangian"`.
- **Slacks.** `ŝ` is `c + Jp` clipped to the row bounds, and
  `sᵢ = clip(cᵢ − πᵢ/ρᵢ)`, or `clip(cᵢ)` when `ρᵢ = 0`.
- **φ′(0)** is computed exactly: it is linear in `ρ`, so LSPρ has the closed
  form `ρ* = ξ·q⁺/‖q⁺‖²`.
- **Line search.** It keeps the Armijo test with constant 10⁻⁴ and step
  halving. SNOPT's safeguarded cubic interpolation is not adopted.
- **BFGS and the convergence test** keep using the QP multipliers.
- **Elastic mode.** Steps whose elastic QP leaves rows violated use the ℓ1
  merit with weight γ, as today. Steps with a consistent linearization use
  the augmented Lagrangian.
- **Restoration** resets `ρ` and `π` to 0, as it resets Powell's penalties
  today.
- **Second-order correction.** The augmented Lagrangian is smooth and needs
  no correction, so the combination is rejected with a clear error instead
  of being given an untested meaning.
- **Option `SqpOptions.violation_limit`** (`None` by default, SNOPT's
  `τ_V = 10` when set):
  - it works with either merit;
  - a trial whose row violation exceeds `τ_V·max(1, violation at x₀)` is
    rejected before any derivative call;
  - rejections are counted in `RunReport.violation_limit_rejections`.

**Validation:**
- φ′(0) against a finite difference of the merit along the path;
- LSPρ closed form and the damping rule against the paper's formulas;
- the Maratos example, where unit steps should be accepted without the
  correction;
- the families with the ℓ1 merit, the ℓ1 merit plus correction, and the
  augmented Lagrangian;
- the violation limit on a model that is undefined far outside the feasible
  region.

## Step 3: Linear constraints first

**SNOPT** (GMS §1.1, §6.1):
- An elastic LP checks the linear constraints before any nonlinear function
  is evaluated.
- A proximal-point problem moves `x₀` onto them:
  - PP1 minimizes `‖x − x₀‖₁`;
  - PP2 minimizes `‖x − x₀‖₂²`.
- All later iterates satisfy the linear constraints, and the merit function
  contains only the nonlinear ones.

**Here:**
- **API.** `ProblemSpec::with_linear_rows(rows)` in Rust, and
  `ProblemSpec(..., linear_rows=[...])` in Python, mark rows whose Jacobian
  row is constant. Undeclared problems behave as today.
- **Start.** The model is evaluated at `x₀`, as today; this is needed for the
  coefficients. If a linear row is violated:
  - the solver solves PP2 with the QP backend (`H = I`, the linear rows and
    the variable bounds);
  - an infeasible PP2 ends the solve with the new status
    `linear_infeasible`. Unlike `linearization_infeasible` this concerns the
    problem itself, since these rows are exactly linear;
  - otherwise the solve starts from the projected point
    (`RunReport.linear_start_projection`).
- **Iterates stay feasible** for the linear rows:
  - every step QP keeps them as hard rows, including the elastic QP. This
    needs per-row hardness in the active-set backend and in the elastic
    KKT check (`QpOptions.hard_rows`);
  - the restoration QP keeps them as hard rows;
  - a convex combination of two points feasible for linear rows is feasible,
    so the line search cannot leave them.
- **Merit and correction.** Linear rows get no merit weight in either merit,
  and no second-order shift.
- **Misdeclared rows.** After every derivative call the declared rows'
  Jacobian entries are compared with the start values. A change beyond
  rounding ends the solve with `InvalidInput`, instead of silently trusting
  a wrong declaration.
- **Families:**
  - the orbit transfer, whose `u(t_f) = 0` row is linear;
  - the MPCC family, whose sum row is linear;
  - a new linear-guard family, where nonlinear functions are undefined (for
    example a logarithm of a linear expression) outside the linear rows.
    This is the trajectory situation of times that must stay ordered.

  Each family is measured with and without the declaration: statuses,
  iterations, callbacks and undefined trials.

## Step 4: Conservative scaling (gap 5)

**SNOPT.**
- The default `Scale option 1` scales the linear constraints and the
  variables (GMS §7.1).
- `Scale option 2` also uses the nonlinear Jacobian at the first point that
  satisfies the linear constraints. The SNOPT manual recommends it only with
  a good start and a mildly nonlinear problem.
- The scaling procedure is the iterative geometric-mean scaling of MINOS
  (Fourer 1982). My description of it is from background knowledge; the
  survey did not check it against a primary source.

**Here:**
- **Option `SqpOptions.scaling`:**
  - `"none"` (default);
  - `"linear"`: rows declared linear, and the variables in them;
  - `"jacobian"`: all rows, from the Jacobian at the first point that
    satisfies the linear rows.
- **Procedure.** Alternating row and column passes divide by
  `√(min·max)` of the absolute nonzeros. There are at most 20 passes, and the
  procedure stops when the largest column ratio improves by less than 10%.
- **Exactness.** Factors are rounded to powers of two, so scaling and
  unscaling are exact in floating point.
- **What is scaled.** The QP and the BFGS matrix live in scaled coordinates:
  - `x = D x̃` and `c̃ = R c`;
  - the step QP becomes `(B̃, Dg, RJD)`, with scaled bounds;
  - `p = D p̃`, `λ = Rλ̃` and `z = z̃/D`;
  - BFGS updates on `(s/D, D y)`;
  - `B̃₀ = I`, which is `D⁻²` in original units;
  - restoration and the proximal point work in the scaled variables;
  - the elastic weight applies per scaled row.
- **What is not scaled.** The merit function, the line search, the
  convergence test and all reported values stay in original units. This
  differs from SNOPT, which tests in scaled units; it keeps the verification
  contract.
- **With factors of 1** every operation is exact, which the bit-identity
  check confirms.
- **Families.** Badly scaled copies of the stress and orbit families, with
  random power-of-ten variable and row scalings, are solved with each
  scaling option. The well-scaled originals check that scaling does no
  harm.

## Deliverables

- Code, tests and Python bindings for the four options. Every option is
  validated before any model call, and every new report field is available
  in Python.
- `scripts/compare_variants.py`. It runs named option sets on all families,
  prints statuses, iterations, callbacks and the paired totals, and fails on
  any false convergence claim.
- `docs/snopt-lessons.md` with the method, contract and measured results of
  each step.
- Updates to `gaps.md`, `api.md` and the README.
- The bit-identity check of the default path.
- A PR against `main`.
