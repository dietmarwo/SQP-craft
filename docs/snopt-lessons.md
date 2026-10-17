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
| 2. Augmented-Lagrangian merit | Maratos example: −25% iterations, −49% callbacks (the second-order correction: −16%, −21%); stress family: −3.7% iterations, −4.9% callbacks, no status changes | complementarity family: +0.4% callbacks with the active-set backend, 9 of 532 problems lost with Clarabel | off |
| 2. Violation limit | a model unbounded outside the constraints: violation ≤ 10 instead of 99 | nothing on the families | off |
| 3. Linear rows | linear-guard family: −7% iterations; elastic mode and restoration never leave the rows | the projection costs two callbacks per projected start (+17% on the MPCC family) | undeclared |
| 4. Scaling (`"jacobian"`) | badly scaled families: orbit 40/40 converged instead of 20/40 with −76% iterations, stress −29% | well-scaled families: +2% to +24% iterations; the orbit converges to another local optimum | off |

None of the four is made the default. Each helps on its own mechanism and
costs elsewhere, and the target application, which would decide, is not
available.

The smooth merit is the strongest candidate for a default. It takes 3.7%
fewer iterations on the stress family with both backends and 25% fewer on
the Maratos example, changes no stress-family status, and makes the
second-order correction unnecessary. Against that, Clarabel loses 9 of the
532 complementarity problems it solves.

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

| Family | ℓ1 merit | ℓ1 + second-order correction | Augmented Lagrangian |
| --- | --- | --- | --- |
| Stress, 1,500 (AS) | 1,485 converged | 1,483 converged | 1,485 converged |
| Stress, the common 1,483 (AS) | 15,128 iterations, 35,897 callbacks | 14,777, 35,565 | 14,567 (−3.7%), 34,121 (−4.9%) |
| Stress, the common 1,483 (Clarabel) | 15,157, 35,978 | 14,818, 35,712 | 14,620 (−3.5%), 34,340 (−4.6%) |
| Complementarity, 600 (AS) | 600; 2,721 iterations, 6,651 callbacks | identical | 600; 2,734, 6,680 |
| Complementarity (Clarabel) | 532 converged | 532 | 523 |
| Maratos, 200 starts | 2,137 iterations, 8,278 callbacks | 1,796, 6,543 | 1,606 (−25%), 4,242 (−49%) |
| 13 SLSQP cases | 344, 737; 13/13 at the reference | 342, 733 | 344, 736; 13/13 |

The smooth merit accepts the unit steps on curved constraints without a second
QP. It does not change the statuses on the stress family, where the
correction loses two problems. On the Maratos example from a feasible start
every step has length 1 (`the_augmented_lagrangian_merit_takes_unit_steps_on_a_curved_constraint`).

**Two corrections to the paper's rules were needed**:
- **The penalty.** With the minimum-norm `ρ*` of (LSPρ), the merit cost the
  stress family 19% more callbacks with the active-set backend, and with
  Clarabel 100 of 1,500 problems ended in `stalled` or `line_search_failure`.
  Near a solution the multiplier estimate `π` lags the QP multipliers by one
  step. That makes the unpenalized slope positive, so `ρ*` is just large
  enough for the slope. But the penalty term falls by only half its slope
  over the unit step, and the unit step was rejected; halving `α` then needed
  about eight trials. The traced case (stress seed 1, case 14, with Clarabel)
  had a violation of 10⁻⁶ and a step of 1.7·10⁻⁷. The merit rose by 5·10⁻¹²
  against a predicted decrease of 2.6·10⁻¹⁴. Requiring the linear-model
  decrease over the unit step, as below, doubles `ρ*` and removes all of
  these failures.
- **Noise-level steps.** A step whose predicted decrease is below the
  rounding level of the merit value is tried at full length (below). With
  the corrected penalty it is rarely needed, but it keeps rounding from
  deciding the line search.

The violation limit changes little on the families. The stress family takes
14,637 instead of 14,661 iterations with the smooth merit, and 335 trial
points are rejected with the ℓ1 merit. Its purpose is the unbounded case
(`the_violation_limit_keeps_trial_points_near_the_constraints`):
- for min −10x₀ on the unit disc from the origin, where `J = 0`, the first
  step goes to x₀ = 10, a violation of 99;
- with `τ_V = 10` every accepted point stays within 10, and the solve needs
  6 instead of 9 iterations.

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

**The penalty.** The paper's (LSPρ) is `min ½‖ρ‖²` subject to
`φ′(0) = −½pᵀBp, ρ ≥ 0`. Here the penalty term `½ρ(1 − α)²r²` matters: over
the unit step it falls by only `½ρr²`, half of what its slope promises. So the
slope must also cover that remainder, `φ′(0) + ½Σρᵢqᵢ = −½pᵀBp`. That makes
the linear-model change of the merit over the unit step `−½pᵀBp`. The
closed form is `ρ* = 2ξq⁺/‖q⁺‖²`, with
`ξ = gᵀp − (π̄ − π)ᵀr − πᵀr′ + ½pᵀBp` (`ρ* = 0` if `ξ ≤ 0`), twice the
paper's. The damped
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
at most three times in a row, as for the ℓ1 merit.

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
  iterations. Declaring the rows without scaling gives 17,046 iterations, so
  almost all of that gain comes from the scaling.
- **`"jacobian"` on the 13 SLSQP cases** converges in fewer than half the
  iterations. The two orbit transfers, however, reach other local optima:
  −1.521507 instead of −1.522060 (N = 20) and −1.524281 instead of −1.524487
  (N = 40). The scaled start `B₀ = D⁻²` takes another path, as the elastic
  weight 10 did ([elastic-mode.md](elastic-mode.md)).
- **Clarabel** shows the same picture on the badly scaled families:
  - stress: 579 instead of 569 converged;
  - linear-guard: 597 with Jacobian scaling, or 598 with the rows declared
    and linear scaling, instead of 588;
  - orbit: 40 instead of 19 converged, with 2,518 instead of 8,206
    iterations.

  On the complementarity family Jacobian scaling makes its
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

## All four together

The configuration closest to SNOPT combines all four:
`hessian_update="augmented"`, `merit="augmented-lagrangian"`,
`violation_limit=10`, declared linear rows and `scaling="jacobian"`
(active-set backend).

| Family | Default | All four | Without the smooth merit and limit |
| --- | --- | --- | --- |
| Stress, 1,500 | 1,485 converged; 15,141 iterations, 35,932 callbacks (common 1,484) | 1,485; 14,957, 35,153 | 1,485; 15,361, 36,751 |
| Complementarity, 600 | 600; 2,721, 6,651 | 600; 2,888, 8,205 | 600; 2,888, 8,205 |
| Maratos, 200 | 2,137, 8,278 | 1,741, 4,615 | 2,217, 8,574 |
| Linear-guard, 600 | 600; 10,121, 23,239 | 600; 12,040, 29,449 | 600; 11,080, 25,923 |
| 13 SLSQP cases | 344, 737; 13/13 at the reference | 161, 406; 11/13 | 172, 420; 11/13 |
| Stress, badly scaled, 600 | 571 converged | 581 (5 line-search failures) | 585 |
| Linear-guard, badly scaled | 600; 17,286, 39,177 | 600; 13,542, 33,706 | 600; 12,554, 29,737 |
| Orbit, 40 badly scaled | 20 converged; 9,944 iterations | 40; 2,349 | 40; 2,458 |

The combination behaves like its parts:
- **Scaling** drives the badly scaled families and costs on the well-scaled
  ones.
- **The smooth merit** drives the Maratos example and the well-scaled stress
  family.
- **The projection** costs on the complementarity family.

No setting is best everywhere, which is why the four stay separate options.

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
- **SNOPT's interpolating line search.** Step halving was kept for both
  merits.
- **Convergence tests in scaled units.**
- **Linear constraints given as data**, which would remove the start's extra
  evaluation and allow a start where the model is undefined.
