# POUNCE curvature pilot

SQP-craft 0.20.0 at `1589f8bc57`, POUNCE 0.12.0, S2MPJ `fea6a70048`. Intel(R) Xeon(R) Processor @ 2.10GHz, Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1.

Solved means independently accepted first-order KKT. LS denotes a recovered least-squares certificate. Supplied duals counts points certified by the solver's own multipliers; the next column gives how many of them lacked multipliers of fixed variables, which the checker took from stationarity. F and D are distinct grouped value and derivative points. H counts analytic Lagrangian-Hessian calls, including failed attempts. The exact callbacks emit a full lower triangle on this panel. The evidence includes solver duals, residuals and model timings.

## Outcomes

| Solver | Runs | Verified | Supplied duals | Of these, fixed variables completed | Recovered duals | Strict and verified | Verified without strict claim | False strict claims | Acceptable-level (rejected) | Objective mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 20 | 20 | 20 | 0 | 0 | 20 | 0 | 0 | 0 (0) | 0 |
| pounce-sqp-exact | 20 | 19 | 18 | 4 | 1 | 18 | 1 | 0 | 0 (0) | 0 |
| pounce-sqp-lbfgs | 20 | 20 | 19 | 5 | 1 | 20 | 0 | 0 | 0 (0) | 0 |
| pounce-sqp-damped | 4 | 4 | 4 | 1 | 0 | 4 | 0 | 0 | 0 (0) | 0 |
| clarabel | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 (0) | 0 |
| clarabel+exact | 4 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 (0) | 0 |

### Terminal solver statuses

- auto — converged 20.
- pounce-sqp-exact — Solve_Succeeded 18, Maximum_Iterations_Exceeded 2.
- pounce-sqp-lbfgs — Solve_Succeeded 20.
- pounce-sqp-damped — Solve_Succeeded 4.
- clarabel — converged 4.
- clarabel+exact — converged 4.

## Paired primary results

**auto / pounce-sqp-exact**

- Both verified: 19. A only: CATENARY (Maximum_Iterations_Exceeded). B only: none. Neither: none.
- Different verified objectives: HS2: 0.050426188 / 4.9412293. Zero first-order work: none.
- On 18 comparable points, median paired F+D ratio 1.518 (A/B). F+D wins and ties 5/11/2 (A/B/tie). Median H counts 0/6.5. Median model callback times 40.5/63.9 ms in the panel.
- Counting actual grouped F+D attempts, median paired ratio 1.518 (18 pairs).
- Restricting to points verified with both solvers’ supplied duals, median paired grouped-attempt ratio 1.518 (18 pairs).
- Restricting further to pairs where neither solver left out a fixed-variable multiplier, 1.121 (14 pairs).

**auto / pounce-sqp-lbfgs**

- Both verified: 20. A only: none. B only: none. Neither: none.
- Different verified objectives: HS2: 0.050426188 / 4.9412293. Zero first-order work: none.
- On 19 comparable points, median paired F+D ratio 0.836 (A/B). F+D wins and ties 11/6/2 (A/B/tie). Median H counts 0/0. Median model callback times 52.3/83.9 ms in the panel.
- Counting actual grouped F+D attempts, median paired ratio 0.836 (19 pairs).
- Restricting to points verified with both solvers’ supplied duals, median paired grouped-attempt ratio 0.836 (19 pairs).
- Restricting further to pairs where neither solver left out a fixed-variable multiplier, 0.833 (14 pairs).

**pounce-sqp-exact / pounce-sqp-lbfgs**

- Both verified: 19. A only: none. B only: CATENARY (Maximum_Iterations_Exceeded). Neither: none.
- Different verified objectives: none. Zero first-order work: none.
- On 19 comparable points, median paired F+D ratio 0.426 (A/B). F+D wins and ties 14/4/1 (A/B/tie). Median H counts 7/0. Median model callback times 50.6/65.9 ms in the panel.
- Counting actual grouped F+D attempts, median paired ratio 0.426 (19 pairs).
- Restricting to points verified with both solvers’ supplied duals, median paired grouped-attempt ratio 0.491 (18 pairs).
- Restricting further to pairs where neither solver left out a fixed-variable multiplier, 0.750 (14 pairs).

## All 20 cases

| Case | n, m | SQP-craft BFGS | POUNCE exact | POUNCE L-BFGS |
|---|---:|---|---|---|
| ROSENBR | 2, 0 | verified, 54/35/0 | verified, 7/7/7 | verified, 84/84/0 |
| ALLINITU | 4, 0 | verified, 18/11/0 | verified, 86/53/53 | verified, 18/18/0 |
| ARWHEAD | 10, 0 | verified, 17/8/0 | verified, 7/7/7 | verified, 17/17/0 |
| BRYBND | 10, 0 | verified, 194/118/0 | verified at Maximum_Iterations_Exceeded, 307/203/500 (1105 F+D attempts) | verified, 159/159/0 |
| HS2 | 2, 0 | verified, 36/18/0 | verified (LS), 8/8/8 | verified (LS), 23/23/0 |
| HS38 | 4, 0 | verified, 44/28/0 | verified, 29/29/29 | verified, 248/248/0 |
| JNLBRNG1 | 25, 0 | verified, 18/13/0 | verified (fixed), 2/2/2 | verified (fixed), 14/14/0 |
| BQPGABIM | 50, 0 | verified, 279/50/0 | verified (fixed), 2/2/2 | verified (fixed), 43/43/0 |
| HS21 | 2, 1 | verified, 3/2/0 | verified, 2/2/2 | verified, 3/3/0 |
| HS44 | 4, 6 | verified, 6/6/0 | verified, 2/2/2 | verified, 6/6/0 |
| HS48 | 5, 2 | verified, 11/9/0 | verified, 2/2/2 | verified, 10/9/0 |
| DTOC1L | 58, 36 | verified, 22/22/0 | verified (fixed), 10/10/10 | verified (fixed), 18/18/0 |
| HS14 | 2, 2 | verified, 6/6/0 | verified, 6/6/6 | verified, 6/6/0 |
| HS71 | 4, 2 | verified, 6/6/0 | verified, 54/6/6 (61 F+D attempts) | verified, 8/8/0 |
| HS78 | 5, 3 | verified, 10/9/0 | verified, 53/5/5 (59 F+D attempts) | verified, 9/9/0 |
| HS100 | 7, 4 | verified, 34/17/0 | verified, 17/9/9 | verified, 33/28/0 |
| POWELLBS | 2, 2 | verified, 13/13/0 | verified, 257/52/52 | verified, 260/55/0 |
| ALSOTAME | 2, 1 | verified, 5/5/0 | verified, 5/5/5 | verified, 6/6/0 |
| CATENARY | 15, 4 | verified, 137/68/0 | Maximum_Iterations_Exceeded, 3/2/2 | verified (fixed), 128/71/0 |
| DTOC2 | 58, 36 | verified, 61/61/0 | verified (fixed), 15/10/10 | verified (fixed), 460/460/0 |

## Controls

| Case | POUNCE damped BFGS | SQP-craft Clarabel BFGS | SQP-craft Clarabel exact | Native exact diagonal shifts | Largest shift |
|---|---|---|---|---:|---:|
| ROSENBR | verified, 57/58/0 (116 F+D attempts) | verified, 54/35/0 | verified, 29/22/22 | 0 | 0 |
| HS44 | verified, 6/7/0 (14 F+D attempts) | verified, 6/6/0 | verified, 6/6/6 | 6 | 2 |
| HS71 | verified, 7/8/0 (16 F+D attempts) | verified, 6/6/0 | verified, 57/57/57 | 57 | 14 |
| DTOC2 | verified (fixed), 73/73/0 | verified, 61/61/0 | verified, 16/12/12 | 5 | 0.178739 |

## Release fixtures

- POUNCE #939 — pass.
- POUNCE #922 — pass.
