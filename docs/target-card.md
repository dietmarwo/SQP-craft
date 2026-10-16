# Application target card (SQP track) — intentionally incomplete

Status: awaiting the Python low-thrust trajectory application. Toy examples
must not fill these fields. See Idea 004 concept v3, "Target contracts and
protection against benchmark overfitting".

| Field | Value |
| --- | --- |
| Application source and data revision | TBD |
| Transcription (shooting/collocation), dimensions `n`, `m`, Jacobian `nnz` | TBD |
| Units and variable/constraint scaling | TBD |
| Optimization framework currently called by the application | TBD |
| Reference solver, version and settings | TBD |
| Saved reference point with independently recomputed residuals | TBD |
| Required objective quality; feasibility tolerances per constraint group | TBD |
| Derivative source (analytic, AD, finite differences) and its check | TBD |
| Admissible starts; stationarity and trajectory (re-integration) checks | TBD |
| Regions where the model is undefined (to exercise `UndefinedPoint`) | TBD |
| Time, evaluation, core and memory budget | TBD |
| End-to-end versus solver-only timing boundary | TBD |
| Callback, copy and GIL-wait accounting | TBD |
| Development, checkpoint and held-out partitions | TBD |
| Failure rate and adverse cases | TBD |
| First experiment hypothesis, time box and stop rule | TBD |

Freeze tolerances and held-out cases before adaptive tuning. Feasibility and
objective quality must be recomputed outside the solver, in unscaled units.
