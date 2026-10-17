# Elastic mode

Version 0.14.0, 16 October 2026. Item 2 of the [order of work](gaps.md#order-of-work),
for gap 4 (inconsistent linearizations handled by a separate restoration
mode). It is opt-in and needs the active-set backend:

```python
solve(spec, model, x0, method="sqp",
      method_options=SqpOptions(qp_solver="active-set", elastic_mode=True))
```

In Rust: `NativeSqp<ActiveSetQp>` with `SqpOptions { elastic_mode: true, .. }`.

## Result

Elastic mode solves every step QP as an ℓ1-elastic QP, as SNOPT does when a
QP is infeasible. The rows may be violated at a cost of γ per unit, so the
QP is always feasible. Feasibility restoration is then never needed:
multipliers and the BFGS matrix carry through infeasible phases, and there
is no mode switch.

It does not make the solver more robust on the test families. The failures
that remain are local, for both strategies:

| Family (active-set backend) | Restoration | Elastic mode |
| --- | --- | --- |
| Stress, 1,500 problems (seeds 1–5) | 1,485 converged; 13 `restoration_failed`, 2 `restoration_limit` | 1,486 converged; 14 `locally_infeasible` |
| The 281 of those that need restoration, both converged (265) | 11.5 iterations including restoration steps, 24.9 callbacks | 10.3 iterations, 25.7 callbacks |
| Complementarity problems, 600 (seeds 1–2) | 600 converged, 4.5 iterations | 600 converged, 4.6 iterations |
| HS071 from (1, 1, 1, 1) | 8 iterations, 1 restoration step, 9 f / 9 ∇f calls | 8 iterations, 1 elastic step, 10 f / 9 ∇f calls |
| Orbit transfer, 84 and 164 variables | converged | the same path, bit for bit |

- **Stress family.** Elastic mode converges on both problems where
  restoration ran out of its step budget. It ends locally infeasible on one
  problem (seed 4, case 23) that restoration solves. The other 13 failures
  are common to both. They end where the projected gradient of the squared
  violation is nearly zero (at most 1.4·10⁻³, with violations from 0.02 to
  8): near local minimizers of the infeasibility, which no local method
  escapes. SLSQP fails 12 of the 14 seen in seeds 1–4, as
  recorded before.
- **No false convergence** in any run. Over the 1,500 stress problems the
  mean iteration count is 10.21 in elastic mode and 10.20 with restoration.

`locally_infeasible` is a new status. It means elastic mode reached a
stationary point of f + γ·violation with violated rows, and stayed there
after six tenfold increases of γ. Like `restoration_failed`, it is not an
infeasibility certificate.

Clarabel was also run on the new complementarity family
(`examples/stress_mpcc.py`). Its QPs there are degenerate: it converges in
531 of 600 problems, against 600 for the active-set backend. It also makes
**one false convergence claim** (seed 1, case 80). The QP multipliers reach
about 5·10⁹, so the stationarity residual is at its own rounding level:
9.5·10⁻⁷ in Rust and 1.2·10⁻⁶ in the NumPy evaluator, against the
tolerance 10⁻⁶. This is independent of elastic mode and is recorded in
[gaps.md](gaps.md).

## Method

### The elastic QP (active-set backend)

```text
minimize   ½pᵀBp + gᵀp + γ Σ_i dist(a_iᵀp, [l_i, u_i])
subject to variable bounds on p (hard)
```

This is the ℓ1-penalty QP with slacks eliminated. Its dual is the ordinary
dual with every row multiplier boxed, `|λ_i| ≤ γ`. The Goldfarb–Idnani
method handles it with three additions:

- **Bounded multipliers.** Each working-set member carries multiplier
  bounds: `[0, ∞)` and `(−∞, ∞)` for hard inequalities and equalities, and
  `[0, γ]` and `[−γ, γ]` for rows. A partial step stops at whichever bound
  is reached first. At 0 the member is dropped, as before. At `±γ` it leaves
  the working set and becomes **elastic**: its multiplier is fixed at `±γ`,
  and that contribution moves into the linear term,
  `g_eff = g − γ Σ σ_e a_e`. A row entering the working set does the same
  when its own multiplier reaches γ before the row is satisfied.
- **Re-entry.** An elastic row can end up strictly satisfied because the
  iterate moves. Its fixed multiplier is then wrong. In slack terms, the
  bound `s ≥ 0` is violated. The row is selected like any violated
  constraint and enters from the opposite side, with bounds `[0, γ]`.
  Together with the elastic term this gives the row a multiplier in
  `σ·[0, γ]` on its bound. If the new multiplier reaches γ, the two cancel
  and the row is simply inactive. The randomized tests exercise this path
  about 26,000 times.
- **Polishing** recomputes `x` and `u` from the factors with `g_eff`. The
  start phase and the final settle release members whose multipliers lie
  outside their bounds: below 0 they are dropped, at `±γ` they become
  elastic.

A row has at most one elastic side, so the state stays small. Hard
constraints are only the variable bounds, and they are always consistent, so
an elastic QP is never infeasible.

### Checking an elastic solution

`QpModel::accepts_elastic` and `kkt::elastic_residuals_with_workspace`
recompute the first-order conditions of the ℓ1 problem from the model data
alone:

- the residuals of the model with each violated row's bounds relaxed to its
  value;
- `max(|λ_i| − γ, 0)` in `dual_sign`;
- `violation_i · |γ − σ_i λ_i|` in `complementarity` (a violated row must
  have `λ_i = ±γ`).

Rows violated by at most the primal tolerance count as satisfied. A first
version applied the elastic test to them too, and a QP solution that sat on
its bound to 7·10⁻¹¹ failed by `γ · 7·10⁻¹¹` (complementarity problem
seed 1, case 167, after the driver moved one step component onto its
bound). A solution the elastic check accepts is optimal, since the problem
is convex.

### The driver

- **Weight.** γ starts at `elastic_weight · max(1, ‖∇f(x₀)‖∞)`. The default
  `elastic_weight` of 10⁴ is SNOPT's default elastic weight. On the
  164-variable orbit, smaller weights leave rows violated in early QPs:
  - with 100, the solve reaches the same optimum in 202 iterations instead
    of 167;
  - with 10, it reaches another local optimum in 143 iterations (objective
    −1.52428 instead of −1.52449).

  On the stress family the counts change by at most two. With 10⁴ the orbit
  QPs never leave a row violated, and the path is identical to the one
  without elastic mode.
- **Merit.** Powell's per-row penalty rule is kept, except that rows the
  elastic QP leaves violated are weighted by exactly γ. The step then
  minimizes the model of the merit, so the predicted decrease is at least
  `pᵀBp > 0`. A weight of 1.1γ could make it negative on a row whose
  linearized violation grows. When the linearization becomes consistent
  again, Powell's rule restarts from the current multipliers, as after
  restoration. Otherwise penalties of γ shorten the following steps: before
  this restart the stress family needed 10.44 iterations on average, after
  it 10.21.
- **Penalty stationarity.** At an infeasible iterate whose QP leaves rows
  violated, the driver evaluates the ℓ1 first-order conditions of the
  nonlinear problem with the QP multipliers. The tolerances are the solver's
  own, with complementarity and sign scaled by `1 + γ`; a step below
  `10⁻¹⁴(1 + ‖x‖)` counts as stationary too. If they hold, γ is multiplied
  by 10 and the QP solved again, up to six times. After that the solve
  stops as `locally_infeasible`.
- **Restoration** stays in place for an `Infeasible` QP outcome, which
  elastic QPs do not produce.
- **Reports:** `elastic_steps` (accepted steps with violated QP rows),
  `elastic_weight` (the final γ, `None` without elastic mode) and
  `elastic_weight_increases`.

## Validation

- **Core** (`crates/core/tests/kkt_residuals.rs`): the elastic residuals
  accept a violated row only at the weight, report a multiplier above it,
  and treat tolerance-level violations as satisfied.
- **Backend** (`crates/activeset/src/tests.rs`):
  - contradicting rows, with both violated at `±γ`;
  - a capped multiplier and a soft equality, both with closed forms;
  - hard variable bounds;
  - a re-entry with a closed form;
  - a large weight reproducing the ordinary solution;
  - Clarabel rejecting the configuration;
  - weight validation.
- **Randomized** (`crates/activeset/tests/against_clarabel.rs`):
  - 1,000 random feasible and infeasible QPs with γ from 10⁻² to 10³ pass
    the independent elastic check, and none is worse than the ordinary
    solution in the elastic objective;
  - 500 feasible QPs with γ above their multipliers reproduce the ordinary
    step;
  - 400 warm starts reach the cold elastic objective.
- **Driver** (`crates/sqp/tests/elastic_mode.rs`):
  - the zero-Jacobian start, where restoration fails and elastic mode
    converges to x = 1;
  - an infeasible problem, which ends `locally_infeasible` after six
    increases, at the expected point and multipliers;
  - the orbit path, bit for bit;
  - backend and option validation before any model call.
- **Python** (`tests/test_elastic.py`):
  - the same cases;
  - HS071 without restoration;
  - the inconsistent restoration fixture, which ends `locally_infeasible`.
- **Scripts:**
  - `examples/stress_random.py --qp-solver active-set --elastic` and
    `scripts/validate_performance_stress.py ... --elastic` (seeds 1–5: no
    errors, no false claims);
  - the new `examples/stress_mpcc.py`, run in CI for the active-set backend
    with and without elastic mode.

## Limits

- Elastic mode changes which local failures occur, not how many. Escaping
  local minimizers of the infeasibility needs global or randomized
  strategies.
- On a problem with redundant nonlinear equalities (`q = 1` and `q² = 1`
  with `q = ‖x‖²`), whose linearizations are inconsistent everywhere off the
  feasible set, every configuration does poorly from 100 random starts:
  elastic mode converges 11 times (mostly running out of callbacks),
  restoration 15 times and Clarabel 9 times.
- The QP time is unchanged when no row is elastic. The elastic check costs
  one extra relaxed-bounds pass per QP.
