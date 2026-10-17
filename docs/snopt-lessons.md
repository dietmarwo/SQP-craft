# Four lessons from SNOPT

Version 0.14.0, 17 October 2026. This document carries out the
[plan](snopt-lessons-plan.md) that follows the comparison with SNOPT in the
literature survey. The reference throughout is Gill, Murray and Saunders,
*SNOPT: An SQP algorithm for large-scale constrained optimization*, SIAM
Review 47 (2005) 99–131
([PDF](https://ccom.ucsd.edu/~optimizers/static/pdfs/siam_44609.pdf)),
cited as GMS.

All four steps are opt-in:

```python
SqpOptions(hessian_update="augmented")          # 1. curvature repair
SqpOptions(merit="augmented-lagrangian",        # 2. smooth merit
           violation_limit=10.0)                #    and violation limit
ProblemSpec(..., linear_rows=[...])             # 3. linear rows kept feasible
SqpOptions(scaling="jacobian")                  # 4. scaling ("linear" too)
```

With every new option at its default, the 9,000 stress runs of
`scripts/validate_performance_stress.py` (seeds 1–5, both QP backends) are
bit-identical to `main` (2288a8e), after each of the four steps.

## Summary

All numbers are from `scripts/compare_variants.py`; families and variants
are described in its docstring. "AS" is the active-set backend. Every
convergence claim in every run passed the independent evaluator.

| Step | Where it pays off | Where it costs | Default |
| --- | --- | --- | --- |
| 1. Curvature repair | Maratos example: −4.6% iterations, −7.4% callbacks | nothing measurable (stress ±0.2%) | off |
| 2. Augmented-Lagrangian merit | MERIT_SUMMARY_GOOD | MERIT_SUMMARY_BAD | off |
| 2. Violation limit | a model unbounded outside the constraints: violation ≤ 10 instead of 99 | nothing on the families | off |
| 3. Linear rows | linear-guard family: −7% iterations; elastic mode and restoration never leave the rows | the projection costs two callbacks per projected start (+17% on the MPCC family) | undeclared |
| 4. Scaling (`"jacobian"`) | badly scaled families: orbit 40/40 converged instead of 20/40 with −76% iterations, stress −29% | well-scaled families: +2% to +24% iterations; the orbit converges to another local optimum | off |

None of the four is made the default: each helps on its own mechanism and
costs elsewhere, and the target application, which would decide, is not
available.

## 1. Curvature repair (`hessian_update="augmented"`)

### Result

How often damping fires was measured first, with the new counters
(`RunReport.bfgs_damped_updates`, `IterationRecord.curvature`). On the stress
family the default damps 1,297 of 15,225 updates (8.5%). Of the last three
updates of the 1,485 converged runs, 220 of 4,455 (4.9%) are damped: damping
does act near solutions.

| Family | Damped (default) | Repaired (`"augmented"`) |
| --- | --- | --- |
| Stress, 1,500 (AS) | 1,485 converged; 15,128 iterations, 35,897 callbacks over the common 1,483 | same statuses; 15,138 iterations, 35,969 callbacks; 1,211 repaired, 107 damped |
| Stress (Clarabel) | 15,254 iterations | same statuses, 15,261 iterations |
| Complementarity, 600 (AS) | 600 converged, 2,721 iterations | 600, 2,726 iterations (3 repaired) |
| Complementarity (Clarabel) | 532 converged | 534 converged |
| Maratos, 200 starts | 2,137 iterations, 8,278 callbacks | 2,038 (−4.6%), 7,668 (−7.4%) |
| 13 SLSQP cases | 344 iterations, 737 callbacks | 340, 729; all reach the reference |

The repair replaces most damped updates near the solution (144 of the last
three updates on the stress family are repaired, 75 still damped), but that
does not shorten the stress runs. It helps where a curved constraint carries
the curvature, as in the Maratos example.

**Deviation from SNOPT.** SNOPT skips the update when no repair exists. A
first version did that, and on the complementarity family 65 of 600 problems
ran into the iteration limit (27,950 iterations instead of 2,721, with
25,779 skipped updates). There `sᵀy < 0.2·sᵀBs` came from a `B` that
overestimated the curvature after a large early multiplier, not from
negative curvature. The steps then ran along a single variable, so the
departure from linearity `x_i y_i` was zero and no repair existed. Skipping
kept the overestimate, and the steps stayed at 8.5·10⁻⁵. Damping shrinks
`B` along `s` instead. The final version damps whenever no repair exists.

### Method

GMS §2.10, second modification. Below the threshold `sᵀy < 0.2·sᵀBs` (the
same trigger as Powell's damping), `y` becomes the gradient difference of the
augmented modified Lagrangian (eqs. 2.7–2.8):

```text
y⁺ = y + (J(x⁺) − J(x))ᵀ Ω d_L,     d_L = c(x⁺) − c(x) − J(x)s
```

`d_L` is the departure from linearity. `Ω = diag(ω)` is the minimum-norm
solution of `aᵀω = β, ω ≥ 0`, with `aᵢ = ((J(x⁺) − J(x))s)ᵢ·d_L,ᵢ` and
`β = 0.2·sᵀBs − sᵀy`; the closed form is `ω = β·a⁺/‖a⁺‖²`. Then
`sᵀy⁺ = 0.2·sᵀBs` exactly. The curvature added comes from the rows that
are curved along `s`; linear rows (`d_L = 0`) contribute nothing.

- **No repair exists** when `a⁺ = 0`, or when `‖Δy‖ > 100·‖Bs‖`; the paper
  gives no number for "ω very large". Powell's damping is then used.
- **The first modification is not transferable.** It evaluates `y` from the
  first feasible point of the QP solve, which the dual Goldfarb–Idnani method
  and Clarabel's interior points do not have.
- **Block mode** applies the same rule per block, with `s` restricted to the
  block and `d_L` from the full step.
- **Scaled coordinates** (step 4): `J` is replaced by `R J D` and `d_L` by
  `R d_L`.

### Contract

- `SqpOptions.hessian_update`: `"damped"` (default) or `"augmented"`.
- `RunReport.bfgs_damped_updates` and `bfgs_modified_updates`, also in the
  default mode.
- `IterationRecord.curvature`: `exact`, `damped`, `modified`, `negligible`
  (displacement below 10⁻¹⁴, no update) or `reset`; `None` without a step.

## 2. Augmented-Lagrangian merit and violation limit

### Result

MERIT_RESULT

### Method

GMS §2.7–2.8.

**The merit function** (eq. 2.3) is

```text
M_ρ(x, π, s) = f(x) − πᵀ(c(x) − s) + ½ Σᵢ ρᵢ (cᵢ(x) − sᵢ)²
```

searched along `(x, π, s) + α(p, π̄ − π, ŝ − s)`:
- `p` is the QP step and `π̄` the QP multipliers;
- `ŝ` is `c + Jp` clipped to the row bounds;
- the slack `s` minimizes `M` for the current `(x, π, ρ)` within the row
  bounds: `sᵢ = clip(cᵢ − πᵢ/ρᵢ)`, or `clip(cᵢ)` when `ρᵢ = 0`.

**The slope.** With `r = c − s` and its rate of change `r′ = Jp − (ŝ − s)`,
the slope is

```text
φ′(0) = gᵀp − (π̄ − π)ᵀr − πᵀr′ − Σ ρᵢ qᵢ,   qᵢ = −rᵢr′ᵢ
```

which is linear in `ρ`.

**The penalty** solves (LSPρ): `min ½‖ρ‖²` subject to
`φ′(0) = −½pᵀBp, ρ ≥ 0`. Its closed form is `ρ* = ξq⁺/‖q⁺‖²`, with
`ξ = gᵀp − (π̄ − π)ᵀr − πᵀr′ + ½pᵀBp` (`ρ* = 0` if `ξ ≤ 0`). The damped
update (eq. 2.4) keeps `ρᵢ` below `4(ρ*ᵢ + Δρ)`, moves it to
`√(ρᵢ(ρ*ᵢ + Δρ))` above, and takes the larger of that and `ρ*ᵢ`. `Δρ`
starts at 1 and doubles whenever the trend of `‖ρ‖` reverses.

**The line search.**
- `π` and `ρ` start at 0.
- The Armijo test uses the constant 10⁻⁴ and step halving; SNOPT's
  safeguarded cubic interpolation is not adopted.
- After the step, `π ← π + α(π̄ − π)`.
- BFGS and the convergence test keep the QP multipliers `π̄`.

**Noise-level steps.** The merit changes only to second order in the
violation. Near a solution its predicted decrease can fall below the rounding
of the merit value itself, so a step whose predicted decrease is below
`100ε(|f| + Σ|πᵢrᵢ| + ½Σρᵢrᵢ²)` is a noise-level step. It is tried once at
full length and accepted if the merit does not rise by more than that level,
at most three times in a row, as for the ℓ1 merit. MERIT_NOISE_NOTE

**Other interactions.**
- **Elastic mode.** Steps whose elastic QP leaves rows violated use the ℓ1
  merit with weight γ, as before. The elastic QP's step is a descent
  direction for that merit.
- **Restoration** resets `π`, `ρ` and `Δρ`.
- **Second-order correction.** The combination is rejected: the smooth
  merit does not have the Maratos effect that the correction addresses.

**The violation limit** (eq. 2.5). With `violation_limit = τ_V`, a trial
point at which some row's violation exceeds
`τ_V·max(1, its violation at the start)` is rejected before its derivatives
are evaluated. That applies to line-search trials and second-order
corrections, with either merit. SNOPT uses `τ_V = 10`, and the start is
taken after a projection (step 3).

### Contract

- `SqpOptions.merit`: `"l1"` (default) or `"augmented-lagrangian"`.
- `SqpOptions.violation_limit`: `None` (default) or `τ_V ≥ 1`.
- `RunReport.violation_limit_rejections`.
- `IterationRecord.max_penalty` is `max ρ̄ᵢ` on augmented-Lagrangian steps.

## 3. Linear rows (`ProblemSpec(..., linear_rows=[...])`)

### Result

Three families have linear rows:
- the linear-guard family `examples/stress_linear.py` (new): 600 problems
  whose model is undefined unless event times stay ordered; the duration rows
  `d_k ≥ δ_k` are linear;
- the complementarity family: the sum row;
- the 13 SLSQP cases: `x₀ + x₁ = 1`, and `u(t_f) = 0` in the orbit transfer.

| Family | Undeclared | Declared |
| --- | --- | --- |
| Linear-guard, 600 (AS) | 600 converged; 10,121 iterations, 23,239 callbacks | 600; 9,404 (−7.1%), 22,518 (−3.1%); 348 starts projected |
| Linear-guard (Clarabel) | 600; 10,125 iterations | 600; 9,408 |
| Linear-guard, elastic mode (AS) | as without | as without |
| Complementarity, 600 (AS) | 600 converged; 2,721 iterations, 6,651 callbacks | 600; 2,697, 7,805 (+17%); all 600 projected |
| Complementarity (Clarabel) | 532 converged | 320 converged |
| 13 SLSQP cases | 344 iterations | 342 |

- **The projection costs two callbacks** per projected start, because the
  coefficients come from the start's own evaluation. On the complementarity
  family that is more than the iterations save.
- **Clarabel on the complementarity family.** The 212 lost problems are
  caused by the projected start points, not by the mechanism. Solving the
  undeclared problems from the same projected starts gives 321 converged.
  Clarabel's degenerate QPs fail more often there: 47 of the 600 projections
  put a complementarity pair at (0, 0).
- **No undefined trials on the linear-guard family**, even undeclared: a QP
  step satisfies a linear row exactly at its end, and the line search moves
  between two points with positive durations. The declaration matters where
  an iterate can leave the rows:
  - **elastic mode**, where a soft linear row may be traded against a
    nonlinear one. On the parabola test (`declared_rows_stay_hard_in_the_elastic_qp`)
    the undeclared elastic step lands where the model is undefined and needs
    5 iterations; the declared one keeps `x = 1` and needs 1;
  - **restoration**, which treated satisfied rows as free;
  - **the start.**
- **The projection is needed.** An experiment that declared the rows but
  skipped the projection lost 64 of the 600 complementarity problems to
  `no_descent`. The merit gives a linear row no weight, which is only right
  when the iterate already satisfies it.

### Method

GMS §1.1 and §6.1.

**At the start.** The model is evaluated at `x₀`, as before; that evaluation
supplies the coefficients. If a declared row is violated by more than the
feasibility tolerance, the proximal-point problem PP2 is solved: `min ½‖p‖²`
subject to the declared rows and the variable bounds, using the QP backend
(in scaled variables with step 4). The solve then starts from `x₀ + p`. If
PP2 is infeasible, the solve ends with the new status `linear_infeasible`.
This concerns the problem itself, provided the declaration is right.

**Afterwards** the declared rows hold at every iterate:
- **Every QP keeps them hard.** In the plain step QP all rows are hard. In
  the elastic QP they are hard through `QpOptions.hard_rows`: the active-set
  backend gives them unbounded multipliers, and
  `QpModel::accepts_elastic_with_hard_rows` and
  `kkt::elastic_residuals_with_hard_rows` check them as ordinary rows. In the
  restoration QP they are added as hard rows.
- **The line search cannot leave them.** A convex combination of two points
  on linear rows stays on them.

**What changes for these rows.**
- They get no weight in the ℓ1 merit and no term in the
  augmented-Lagrangian merit (SNOPT's merit has only the nonlinear rows).
- The second-order correction does not shift them, and the elastic test
  never counts them as elastic.

**Misdeclared rows.** After every derivative call, the declared rows'
Jacobian entries are compared with those at the start. A change beyond
`10⁻⁹(1 + |a|)` ends the solve with `InvalidInput`.

**Limit.** SNOPT never evaluates the nonlinear functions before the linear
constraints hold, because it receives them as data. Here the start must be
defined for the model.

### Contract

- `ProblemSpec::with_linear_rows(&rows)` in Rust, `ProblemSpec(...,
  linear_rows=None)` in Python, and the `linear_rows` getter. Pickling keeps
  them.
- `Status::LinearInfeasible`, `"linear_infeasible"` in Python.
- `RunReport.linear_projection`.
- `QpOptions.hard_rows` (empty, or one flag per row) and the two checks
  above. A flag vector of the wrong length is `InvalidInput`.

## 4. Scaling (`scaling="linear"` or `"jacobian"`)

### Result

The badly scaled families take the stress problems, the linear-guard
problems and 20 copies of each orbit transfer, and express them in units
`x = S x′`, `c′ = T c`. `S` and `T` are random powers of ten in
[10⁻², 10²]. The tolerances stay absolute in the new units.

| Family (AS) | `"none"` | `"jacobian"` |
| --- | --- | --- |
| Orbit, 40 badly scaled copies | 20 converged, 20 iteration limit; 9,944 iterations | 40 converged; 2,346 iterations |
| Orbit, the 20 that converge both ways | 3,944 iterations, 8,181 callbacks | 998, 2,488 (−75%, −70%) |
| Stress, 600 badly scaled | 571 converged | 585 converged |
| Stress, the 566 common | 9,932 iterations, 24,500 callbacks | 7,088, 17,308 (−29%, −29%) |
| Linear-guard, 600 badly scaled | 600 converged; 17,286 iterations | 598 (2 line-search failures); 13,183 |
| Stress, well scaled | 1,485 converged; 15,141 iterations (common 1,484) | 1,485; 15,454 (+2.1%) |
| Complementarity, well scaled | 2,721 iterations | 3,364 (+24%) |
| Maratos, well scaled | 2,137 iterations | 2,290 (+7%) |
| Linear-guard, well scaled | 10,121 iterations | 11,712 (+16%) |
| 13 SLSQP cases | 344 iterations, 13/13 at the reference | 160 iterations, 11/13 |

Results by option:
- **`"linear"`** is the identity without declared rows, and gives exactly the
  unscaled runs. On the badly scaled linear-guard family, with the duration
  rows declared, it reaches 600 of 600 in 12,022 instead of 17,286
  iterations (LINEAR_ONLY_NOTE).
- **`"jacobian"` on the 13 SLSQP cases** converges in fewer than half the
  iterations. The two orbit transfers, however, reach other local optima:
  −1.521507 instead of −1.522060 (N = 20) and −1.524281 instead of −1.524487
  (N = 40). The scaled start `B₀ = D⁻²` takes another path, as the elastic
  weight 10 did ([elastic-mode.md](elastic-mode.md)).
- **Clarabel** shows the same picture on the stress and linear-guard families
  (CLARABEL_SCALED). On the complementarity family Jacobian scaling makes its
  degenerate QPs fail more often: 405 instead of 532 converged. The
  active-set backend solves all 600 under every option.

### Method

SNOPT's `Scale option 1` scales the linear constraints and the variables;
`Scale option 2` also uses the nonlinear Jacobian at the first point that
satisfies the linear constraints (GMS §7.1 and the SNOPT manual). The scales
come from MINOS's iterative geometric-mean scaling (Fourer 1982); its details
here are from the manuals, not checked against the code:
- alternate row and column passes divide each row and column by `√(min·max)`
  of its absolute nonzeros;
- at most 20 passes are made, and the iteration stops when a pass reduces
  the largest column ratio `max|ã|/min|ã|` by less than a factor 0.9;
- `"linear"` uses the declared rows only, and `"jacobian"` every row, at the
  start or its projection;
- factors are rounded to powers of two, within `2^±60`.

With `x = D x̃` and `c̃ = R c`:

- **The step QP** is solved in the scaled data `(B̃, Dg, RJD)` and scaled
  bounds, and mapped back with `p = D p̃`, `λ = R λ̃` and `z = z̃/D`. All
  maps are exact for powers of two. The snap of tolerance-level overshoots
  uses `tolerance/D_j`.
- **The BFGS matrix** lives in scaled coordinates: `B̃₀ = scale·I`, which is
  `D⁻²` in original units, and the updates use `s/D` and `D y`.
- **Restoration** solves its Gauss–Newton model in `x̃`: `J D`, with the
  regularization and the trust bound on `p̃`. The proximal point minimizes
  `‖p̃‖`.
- **Elastic mode.** The weight γ applies per scaled row, so row `i` costs
  `R_i γ` per original unit. The merit weight and the penalty-stationarity
  test (`kkt::elastic_residuals_with_row_weights`) use these weights.
- **Unchanged.** The merit, the line search, the violation limit, the
  convergence test and every reported value stay in original units. SNOPT
  tests in scaled units; this keeps the verification contract.

### Contract

- `SqpOptions.scaling`: `"none"` (default), `"linear"` or `"jacobian"`.
- `RunReport.variable_scales` and `row_scales`: the factors, empty without
  scaling.
- With scaling, `report.qp_trace` holds the scaled QPs.

## Validation

- **Rust unit tests:**
  - the repair formula in closed form, the damping fallback, and block
    columns (`bfgs.rs`);
  - the slope against a finite difference, LSPρ and the damped update, the
    slacks, and skipped rows (`merit.rs`);
  - scale factors, unselected rows and names (`scaling.rs`);
  - hard elastic rows (`activeset/src/tests.rs`, `core/tests/kkt_residuals.rs`);
  - linear-row validation (`core/tests/spec_validation.rs`).
- **Driver tests** (`crates/sqp/tests/snopt_lessons.rs`), each step with both
  backends where they apply:
  - the repair on the Maratos example, and the exact damping fallback with
    only linear rows;
  - unit steps of the smooth merit;
  - the violation limit on a model unbounded outside its constraint;
  - option validation before any model call;
  - elastic steps under the smooth merit;
  - the projection and `linear_infeasible`;
  - hard rows in the elastic QP and in restoration;
  - a misdeclared row;
  - Jacobian scaling of the Maratos example in bad units, with original-unit
    residuals;
  - linear scaling as the identity;
  - scaling with elastic mode and with declared rows.
- **Python** (`tests/test_snopt_lessons.py`): the same through the binding,
  plus pickling of linear rows.
- **Scripts:** `scripts/compare_variants.py` produced every table above and
  fails on a false convergence claim; `examples/stress_linear.py` runs the
  new family on its own.

## Not adopted

- **SNOPT's first curvature modification**, which needs a primal QP method.
- **Skipping the update** when no repair exists; see step 1.
- **SNOPT's interpolating line search.** Its effect on the extra trials of
  the smooth merit is untested.
- **Convergence tests in scaled units.**
- **Linear constraints given as data**, which would remove the start's extra
  evaluation and allow a start where the model is undefined.
