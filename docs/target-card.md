# Application target card for structured SQP

The original Idea 004 Python application is still an open validation target.
This card uses the Earth–Mars case from SQP5 as a reproducible stand-in.
The comparison concerns the fixed 300-day Sims–Flanagan discretization and
a local first-order result.

| Field | Value |
| --- | --- |
| Source and data | SQP5 Earth–Mars case at SQP-craft `3b053ec`, Earth ephemeris at MJD2000 9800, Mars 300 days later, pykep-core-rust `754701b` plus `SimsFlanaganChain` (pykep-rust 0.1.6) |
| Dynamics | Central-body Kepler drifts and midpoint impulses, maximum thrust 0.12 in model units, exhaust velocity 30,000 m/s, initial mass 1,000 kg |
| Decisions | Three departure-excess components, four three-component throttles per arc, seven states per interior node, and terminal mass |
| Sizes | 7, 13, 26, 51, 101 arcs give 130, 244, 491, 966, 1,916 variables. A 263-arc run gives 4,994 variables |
| Rows and Jacobian | Per arc seven matching rows and four throttle norms. The 101-arc case has 1,111 rows and 19,524 Jacobian entries |
| Units | AU, solar gravitational time unit and 1,000 kg. KKT is checked in the model's original nondimensional coordinates. DOP853 replay reports km, m/s and kg |
| Objective and bounds | Minimize negative terminal mass, with the original throttle ceiling and departure and arrival-mass bounds. Internal nodes have broad finite boxes. The primary model has a 0.02 throttle floor |
| References | SciPy SLSQP on both formulations at 7 and 13 arcs. Uno 2.9.3 filterSQP and IPOPT presets with L-BFGS on the chain. Reference points are saved with separately recomputed residuals |
| Acceptance | Absolute primal ≤ 1e-7, stationarity and complementarity ≤ 1e-6, dual-sign residual ≤ 1e-8. DOP853 replay uses relative tolerance 3e-12 and absolute tolerance 3e-13 |
| Derivatives | Analytic arc transition Jacobians from pykep, assembled in CSR. Check every decision coordinate with steps 1e-5 and 1e-6, absolute tolerance 2e-7 and relative tolerance 2e-5 |
| Start | Lambert departure and distributed arrival correction. Lift node states by propagating from each endpoint to the global cut |
| Domain | Non-positive mass, zero radius, and propagation failure are outside the valid model. The throttle norm is nonsmooth at a zero throttle |
| Budgets | One numerical thread, 1,500 SQP iterations, 12,000 callbacks, 30 s per solve, 64 MiB estimated structured storage. A separate 90 s run probes 4,994 variables |
| Timing boundary | End-to-end SQP wall time includes Python callback conversion and excludes reference replay and independent post-check. The result reports QP setup, engine and callback phases |
| Case partition | Development on Lambert starts at 7, 13 and 26 arcs. Larger Lambert starts characterize scaling. Offset starts at 51 and 101 arcs are the held-out start cases |
| Adverse cases | Run a floor-off pair with the same start, and a Jacobian-scaled run at each size. Record early QP rejection and stationarity misses as distinct failures |
| Stop rule | Revisit curvature if partitioned needs over 2× the dense iterations on matched converged runs. Examine exact Hessians if steps still rise at fixed arc width |

## Matched single-leg reference

The chain keeps the same global forward/backward cut as the single leg,
including its backward impulse mass convention. At 13 arcs (52 throttle
segments), SLSQP returned terminal mass 900.1501816079 kg for the single
leg and 900.1501816083 kg for the chain. Both points passed the independent
first-order check. The same comparison at 7 arcs agreed to better than
`1e-6` kg. This verifies the discretized objective for these local solves.

The independent DOP853 arc defects agree with the pykep defects to under
`7e-15` in normalized coordinates. Global replay position defects of the
accepted SQP points are below 0.6 m on the tested sizes. They are larger
than the millimeter-scale defects of the small SLSQP references. The latter
used SciPy's tighter solve settings and do not define the SQP KKT thresholds.

## Floor and conditioning

The recorded SQP5 active-set runs took 316 and 950 iterations at 20 and
40 segments after the floor was applied. In the new paired Clarabel runs
with full BFGS and `scaling="jacobian"` on the same start, at 7 arcs the
dense chain converged in 530 steps with the floor and failed its QP after
293 steps without it. At 13 arcs it failed with and without the floor.
This comparison identifies a robustness effect and does not separate the
floor's iteration cost from the choice of merit, scaling and QP backend.
Without scaling, full BFGS with the floor converged in 74 steps at 7 arcs
and 106 at 13 arcs. The factor and iteration measurements are in
[structured SQP](structured-sqp.md).
