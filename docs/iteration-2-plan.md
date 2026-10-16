# Iteration 2 plan — bounded feasibility restoration

16 October 2026 UTC. Baseline: `iteration-1` at
`e627721e73aad429ae6e9df7615c37f892ce6d7e` (Review iteration 1: fix merit penalty, QP acceptance, derivative checker).
`main` still points to the interface draft. Work starts from the reviewed branch;
none of its merit, QP acceptance, derivative-checker or exception-handling fixes
will be replaced by the older ZIP.

Implementation completed: see [iteration-2-results.md](iteration-2-results.md) and
[validation.md](validation.md) for measured outcomes.

## Review and choice

The current implementation has real dense BFGS SQP, per-row Powell merit
penalties, a scale-aware independently checked Clarabel adapter, bounded grouped
callbacks, typed Rust/Python interfaces, iteration history and an orbit-transfer
fixture up to 164 variables. The checked-in report records 57 Rust tests, one
doctest and 67 Python tests. These counts will be verified locally.

One failure is both reproducible and actionable: HS071 from `[1,1,1,1]` has
inconsistent linearized rows although the nonlinear model is feasible. The
current solver returns `linearization_infeasible` without taking a step. This
iteration addresses that failure. The real user trajectory model remains
unavailable; no application speedup or production-ready claim is planned.

## Scope and algorithm

Add an optional, bounded feasibility-restoration path, enabled by default with
`max_restoration_steps=10`; setting zero reproduces the old stop behavior. Enter
only when the normal SQP backend reports an infeasible linearization. After each
accepted restoration step retry the normal SQP subproblem.

Define signed row residuals `r_i = c_i - clamp(c_i, lower_i, upper_i)` and
`psi(x) = 0.5 * sum(r_i^2)`. Equality rows participate even at zero residual;
violated inequality rows participate, inactive inequalities have zero residual.
The restoration direction solves the bound-constrained Gauss–Newton QP

```
min_p 0.5 p^T (J_active^T J_active + delta I) p + (J_active^T r)^T p
subject to max(lower_x-x, -Delta) <= p <= min(upper_x-x, Delta)
```

Defaults: `delta=1e-4` (`restoration_regularization`), `Delta=1`
(`restoration_step_limit`). They are positive finite typed options. There are
no additional slack variables and no change to the 256-variable cap. All
intermediate sums/products must be finite. This is a local least-squares
restoration pilot, not a general feasibility method or an exact penalty theorem.

Independently validate the restoration QP with the existing acceptance rule.
Backtrack on `psi`, with Armijo constant `1e-4` and predicted decrease
`-gradient_psi^T p`. Trials must strictly reduce `psi` and remain strictly in the
variable box. Fixed-variable roundoff corrections use the existing counted,
rechecked policy. Never clip a user start. Only a complete, finite grouped
functions/derivatives evaluation can become the next accepted iterate.

After restoration, reset B to its initial diagonal and clear merit penalties
and stale NLP multipliers/residuals. Do not use restoration-QP multipliers as
NLP multipliers or declare convergence during restoration. The next ordinary
QP and the unchanged original-unit NLP KKT tests decide convergence.
Restoration resets have their own counter, separate from BFGS failure resets.

## Controls, reporting and compatibility

- Accepted restoration steps count toward the existing outer iteration limit;
  all callbacks count toward the shared callback budget. Time, cancellation,
  observer and exception behavior are unchanged.
- Zero `max_restoration_steps` preserves `linearization_infeasible`. Exhausting
  an enabled restoration budget returns `restoration_limit`. Failure to obtain
  descent or an acceptable trial returns `restoration_failed`, never a claim
  that the nonlinear problem is infeasible.
- Record restoration attempts, accepted steps and inclusive elapsed time. With
  `record_history`, keep separate typed restoration records containing before/
  after objective, primal violation and squared-violation merit, step length,
  trial count and QP iterations. Existing SQP history contains normal iterations;
  iteration numbers span both kinds of accepted steps. QP counters include both.
- Keep the public QP trace scoped to ordinary SQP step models. Document that
  restoration time overlaps callback and QP timing categories.
- Update Rust exports, Python dataclasses/status typing/binding together. Bump
  pre-release package version to 0.3.0; retain default `method="dummy"`.

## Work packages and acceptance

1. Verify the branch and baseline; save the exact SHA and baseline checks.
2. Implement a separate private `restoration.rs`, small driver dispatch changes,
   typed controls/statuses/reports and Python conversion.
3. Test HS071 from the previously failing start through Rust and a clean Python
   wheel. Require independent primal/KKT/objective acceptance, not merely a new
   status. Test disabled restoration against the old behavior.
4. Test zero-Jacobian failure, truly inconsistent constraints, fixed variables,
   finite arithmetic, restoration step limits, partial evaluation budgets,
   observer stops, cancellation, undefined trials and original fatal exceptions.
5. Preserve the existing analytic, HS071 and orbit-transfer regression gates.
   Run the unchanged comparison evaluator against the same starts and SLSQP.
   Include every failure; do not weaken tolerances to make the new case pass.
6. Run rustfmt, Clippy, debug Rust tests (overflow checks on), Rust 1.85 check,
   docs, clean-wheel Python tests and dummy-only feature verification. Update CI
   if a new test/benchmark is not already included in its commands.
7. Deliver complete source ZIP, this plan, results, reproducible commands and
   dependency lockfile. No remote push or release publication is part of this task.

## Deferred and known failure modes

No second-order correction, QP workspace/warm-start reuse, sparse Hessian,
automatic scaling, parallel solver, native factorizer or additional backend.
An infeasible stationary point of `psi` (for example `x^2=1` at `x=0`) cannot
be escaped by this first-order local model. Truly inconsistent constraints must
terminate without a feasibility or optimality claim. Poor scaling and dense
`J^T J` conditioning remain limitations; report numerical failure honestly.
The next choice after this iteration should use measured restoration failures
or the existing orbit example's QP-time profile.
