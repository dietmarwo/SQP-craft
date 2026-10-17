# Gaps compared with mature SQP codes

16 October 2026, version 0.14.0 (the reviewed performance release). The
comparison is with SNOPT, filterSQP, NLPQL and SLSQP, and for
second-derivative use also with the IPOPT, WORHP and KNITRO family. It is
ranked by expected impact on the target use, trajectory transcriptions.
Measurements come from this repository's fixtures; statements about other
codes describe their published methods, not runs made here.

## 1. QP subproblems: cold interior point on a dense Hessian

Every step QP is solved from scratch by Clarabel's interior-point method.

- **Measured:** about 8–9 interior-point iterations per QP, each factorizing a
  KKT system that contains the dense n × n BFGS block. That is 85–93% of run
  time on the orbit fixture, about 1.1 ms per factorization at n = 164
  (`docs/iteration-3-results.md`, `docs/performance/README.md`). Workspace
  reuse removes only the setup.
- **Others:** active-set QP solvers (SQOPT in SNOPT, the LSQ solver in SLSQP)
  start from the previous working set. Near convergence that set rarely
  changes, so a QP often costs one or two factor updates.
- **Side effects:** interior-point solutions stop slightly short of active
  bounds. That causes the noise-level predicted-decrease cases handled in
  iteration 2, and no clean active set is ever available.
- **Status:** addressed by the opt-in dense active-set backend
  (`qp_solver="active-set"`, [active-set-qp.md](active-set-qp.md)). On the
  fixtures it takes the same steps, the QP time falls 6.0–7.7×, and it gives
  exact working sets, which warm-start the next QP. Clarabel is still the
  default.

## 2. Curvature: dense full-space damped BFGS

- **Measured:** O(n²) memory and O(n³) work per QP, hence the 256-variable
  cap. The curvature is learned from scratch: 167 major iterations for 164
  variables, about the same as SLSQP's 180 with the same approach.
- **Others:** SNOPT uses limited-memory updates. IPOPT, WORHP and KNITRO can
  use exact sparse Hessians. Collocation problems have very sparse
  Hessians, and with exact second derivatives they typically converge in a
  few dozen iterations, independent of n.

## 3. Globalization: L1 merit, backtracking, no second-order correction

On curved equality constraints near a solution, an L1 merit line search can
reject or shorten good steps (the Maratos effect). SNOPT uses an
augmented-Lagrangian merit function; filterSQP uses a filter with trust
regions. Neither a second-order correction nor a watchdog is implemented
here.

## 4. Inconsistent linearizations: separate restoration mode

The bounded Gauss–Newton restoration is a separate mode that can stall at
infeasible stationary points of the constraint violation.

- **Measured:** 14 of 1,200 stress problems fail this way (SLSQP fails 12 of
  the same 14).
- **Others:** SNOPT's elastic mode puts an ℓ1 penalty on the constraints
  inside the QP itself, so the subproblem is always feasible and the
  transition is smooth.
- **Status:** elastic mode is available as an option
  (`elastic_mode=True` with the active-set backend,
  [elastic-mode.md](elastic-mode.md)). It changes which local failures
  occur, not how many:
  - on the stress family it ends locally infeasible on 14 of 1,500 problems,
    against 15 failures with restoration, 13 of them the same;
  - on the problems that need restoration it takes fewer iterations but
    slightly more callbacks.

  The remaining failures are near local minimizers of the infeasibility,
  which no local strategy escapes. Restoration stays the default.

## 5. No automatic scaling

Tolerances are absolute and in original units. The orbit fixture is well
scaled. Real trajectory models, with times, masses and angles in very
different units, usually are not. SNOPT and IPOPT scale variables and
constraints by default.

## 6. Thin external evidence

The only external comparison is SciPy SLSQP: 13 cases, one random problem
family and the orbit transfer. The mature codes are tuned on CUTEst-scale
test sets, and the failure profile on degenerate, badly scaled or large
problems here is unknown. Some weaknesses may not be on this list yet.

A first addition beyond SLSQP's families: 600 random problems with
complementarity constraints (`examples/stress_mpcc.py`), where the
Mangasarian–Fromovitz condition fails at every feasible point.
- The active-set backend solves all 600, with or without elastic mode.
- Clarabel solves 531. It ends 62 with `qp_failure`, 5 with
  `numerical_failure` and 1 with `line_search_failure`.
- Clarabel also makes one false convergence claim (seed 1, case 80): its QP
  multipliers reach 5·10⁹, so the stationarity residual is at its own
  rounding level (9.5·10⁻⁷ in Rust, 1.2·10⁻⁶ in the independent NumPy
  evaluator, against a tolerance of 10⁻⁶).

The convergence test should not accept a residual below its own rounding
level. That is a separate fix.

## What to keep

These properties should survive every change:

- independent original-unit KKT verification of every convergence claim;
- statuses that never claim more than was shown (no infeasibility claim
  from restoration, no optimality certificate);
- reproducible evidence for every change.

## Order of work

1. **A dense, warm-startable dual active-set QP** (Goldfarb–Idnani) on the
   Cholesky factor of B, for n ≤ 256. It addresses gap 1 directly, gives
   exact active sets, and folds the positive-definiteness check into the
   factorization the method needs anyway. Done as an opt-in backend:
   [active-set QP](active-set-qp.md). Making it the default awaits evidence
   from the target application.
2. Elastic mode on top of the active-set solver (gap 4). Done as an option:
   [elastic mode](elastic-mode.md). It keeps multipliers and curvature
   through infeasible phases but does not reduce the local failures.
3. A second-order correction or watchdog (gap 3).
4. Automatic scaling, measured on the target application once its data
   exists (gap 5).
5. Sparse or limited-memory curvature with a sparse QP when the application
   needs more than a few hundred variables (gap 2).
6. A CUTEst subset through a Python interface, to measure gaps 3–5 instead
   of reasoning about them (gap 6).
