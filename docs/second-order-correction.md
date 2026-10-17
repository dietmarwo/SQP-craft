# Second-order correction

Version 0.14.0, 17 October 2026. Item 3 of the [order of work](gaps.md#order-of-work),
for gap 3: the ℓ1 merit line search rejects good steps on curved constraints
(the Maratos effect). It is opt-in and works with both QP backends and in
elastic mode:

```python
SqpOptions(second_order_correction=True)
```

## Result

`scripts/compare_corrections.py` (active-set backend) gives:

| Family | Without | With the correction | Corrections tried / taken |
| --- | --- | --- | --- |
| Maratos example (N&W 15.4), 200 random starts | 200 converged, 2,137 iterations, 8,278 callbacks | 200 converged, 1,796 iterations (−16%), 6,543 callbacks (−21%) | 607 / 251 |
| Stress family, 1,500 problems | 1,485 converged | 1,483 converged | 1,092 / 521 |
| Stress, the 1,483 that converge both ways | 15,128 iterations, 35,897 callbacks | 14,777 iterations (−2.3%), 35,565 callbacks (−0.9%) | |
| Complementarity family, 600 problems | 600 converged | identical (no correction is tried) | 0 / 0 |
| The 13 SLSQP-comparison cases | | identical, except orbit-20: 104 → 102 iterations | |

- **No false convergence** in any run.
- **Clarabel** gives the same picture (`--qp-solver clarabel`):
  - on the stress family 1,093 corrections are tried and 521 taken, with
    the same two status changes;
  - the 1,483 problems that converge both ways need 14,818 instead of
    15,157 iterations;
  - the Maratos numbers are identical, and the complementarity family is
    unchanged.
- **Two stress problems change status.** They take another path into
  restoration, which fails there. No problem changes the other way. That is
  path dependence of a local method, so the correction stays off by default.
- **Where it pays off.** In seeds 1–2 of the stress family:
  - 4 corrections were tried near a solution (`‖p‖∞ < 10⁻³` or a violation
    below 10⁻⁴), and all 4 were accepted;
  - the other 477 came at the start (median iteration 1, median `‖p‖∞` 2,
    median violation 3), and only 175 of those (37%) were accepted;
  - each failed correction costs one function call.

  The correction is worth having for curved equality constraints reached
  from near-feasible points, as in collocation.

## Method

Nocedal & Wright, Algorithm 18.3, once per iteration:

- **When.** The full step `p` (α = 1) fails the Armijo test on the merit,
  and three more conditions hold:
  - it does not reduce the constraint violation, `θ(x + p) ≥ θ(x)`;
  - the constraints are curved along `p`, meaning
    `‖c(x + p) − c(x) − Jp‖∞` exceeds the feasibility tolerance (for linear
    rows the corrected QP would return the same step);
  - the step is not a noise-level step.
- **The corrected QP.** The step QP is solved again with its row bounds
  shifted by the second-order term: `c(x) + Jp̃` is replaced by
  `c(x + p) + J(p̃ − p)`. The step must then also absorb the curvature of
  the constraints along `p`. Same `B`, gradient and variable bounds; the
  working set of the step QP is the warm start. The solution is snapped to
  its bounds and must pass the same independent check (the elastic one in
  elastic mode).
- **The test.** `x + p̃` is accepted if it passes the Armijo test of the
  full step, `φ(x + p̃) ≤ φ(x) − 10⁻⁴·predicted(p)`. Otherwise the line
  search backtracks along `p` as before, from α = ½. A failed corrected QP
  just skips the correction.
- **Multipliers.** BFGS and the next convergence test keep the multipliers
  of the original step QP. The update uses the actual displacement
  `s = x⁺ − x`.

## Contract

- `SqpOptions.second_order_correction` (default `false`).
- `RunReport.second_order_corrections`: corrected points evaluated.
- `RunReport.second_order_steps`: corrections accepted.
- `IterationRecord.second_order_correction`: a correction was tried at
  this iterate. It was taken when the record's `step_length` is 1.
- Each corrected QP counts in `qp_solves` and the QP timers. Each corrected
  point counts as a trial and as a `functions` call.

## Validation

- `crates/sqp/tests/second_order_correction.rs`:
  - the Maratos example from a feasible start reaches (1, 0) with fewer
    function calls;
  - the accepted correction is flagged in the history;
  - the same holds with Clarabel and in elastic mode;
  - nothing is tried without the option or for linear rows.
- `tests/test_second_order.py`: the same through Python, for both backends.
- `scripts/compare_corrections.py`: the table above, failing on any false
  convergence claim.
- With the option off, the 9,000 runs of `validate_performance_stress.py`
  (seeds 1–5, both backends) are bit-identical to the previous commit.

## Not done

A watchdog (non-monotone acceptance) would also address the Maratos effect,
without the extra QP. A filter or an augmented-Lagrangian merit, as in
filterSQP and SNOPT, is a larger change to the globalization. Repeated
corrections, as in IPOPT, were not tried, since a single correction already
succeeds near solutions.
