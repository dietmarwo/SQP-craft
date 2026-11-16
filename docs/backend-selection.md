# Automatic backend selection

Since 0.19.0, Python's `method="sqp"` defaults
to `SqpOptions(qp_solver="auto", elastic_mode=None)`. The selected Hessian
mode determines the backend; there is no timing heuristic or retry with a
different solver.

| Request | Resolved backend | Omitted / `None` elastic setting |
|---|---|---|
| `auto`, full BFGS or explicit uniform blocks | active-set | enabled |
| `auto`, partitioned BFGS or exact curvature, at any dimension | Clarabel | disabled |
| `active-set`, explicitly selected | active-set | disabled (legacy behavior) |
| `clarabel`, explicitly selected | Clarabel | disabled (legacy behavior) |

An explicit `elastic_mode=False` disables elastic steps. An explicit `True`
requires active-set; with Clarabel it raises an error before model evaluation.
An automatic structured request with `elastic_mode=True` also raises an error;
it does not switch to a dense backend. Explicit backend choices are never
silently replaced, including when a requested mode is unsupported.

Full and uniform-block BFGS keep the 256-variable cap and any smaller
`dense_size_limit`. `auto` does not opt into structured curvature or expand
that cap. Above the dense limit, select `hessian="partitioned"` or `"exact"`
and provide the required structure. The native active-set implementation
still cannot solve these large structured problems. Small structured problems
also choose Clarabel under `auto`; explicit active-set remains available
where supported. Hessian fallbacks do not change the selected backend.

## Python

```python
# Dense default: active-set with elastic steps.
result = solve(spec, model, start, method="sqp")
print(result.report.qp_solver, result.report.elastic_mode)

# Previous Python default, explicitly pinned.
legacy = SqpOptions(qp_solver="clarabel", elastic_mode=False)

# Automatic backend, but keep restoration instead of elastic steps.
restoration = SqpOptions(elastic_mode=False)

# Sparse structured mode: automatic Clarabel, elastic steps off.
structured = SqpOptions(hessian="partitioned", qp_reuse_workspace=True)
```

`report.qp_solver` is the resolved name (`active-set` or `clarabel`);
`report.elastic_mode` is a concrete boolean. Both are present even when
cancellation or a budget stops the solve before its first QP.
`report.qp_backend` remains the numerical backend's versioned diagnostic,
which can be empty if no QP ran. Dummy reports have `qp_solver=None` and
`elastic_mode=False`. `solve` still requires `method="sqp"` to optimize;
its default method remains the dummy evaluator.

Clarabel workspace reuse remains opt-in. It has no effect on active-set;
set `qp_solver="clarabel"` when testing reuse on dense models.

## Jobs and Rust

Jobs still require the `backend` field. `sqp.solve/1` accepts `"auto"`,
`"active_set"` and `"clarabel"`; the underscore spelling is unchanged.
An omitted or null `elastic_mode` follows the same table. Validation replaces
`auto` and the optional elastic setting with concrete values in the resolved
payload, and outcomes report them in `details.options`. Revalidating a resolved
job leaves those choices unchanged. Old jobs with an explicit backend and
no elastic setting retain their nonelastic behavior. Generated examples use
`auto`; large examples also explicitly request partitioned curvature. A
comparison that varies only `backend` between `auto` and an explicit backend,
as an experiment-craft design may, also changes the elastic setting; set
`elastic_mode` in every arm to compare the backends alone.

`NativeSqp<B>` remains generic: Rust callers choose the backend type and
`SqpOptions::default()` keeps `elastic_mode=false`. No new backend dependency
is added to the solver crate. Frontends can use the shared
`QpSolver::resolve(hessian, Option<bool>)` helper and apply **both** returned
fields when constructing their driver. Python and the job worker use this
same helper.

## Evidence and migration

The historical [CUTEst report](benchmark-results.md) motivates this policy:
754 independently accepted problems with Clarabel, 799 with active-set and
813 with active-set plus elastic mode in 0.14; the 0.16 active-set rerun
reached 801. These counts are not a new full-suite benchmark of this change.
Backend selection changes neither numerical tolerances nor KKT checks.

A targeted [53-problem panel](benchmarks/backend-selection/README.md) of
backend disagreements and regressions checks the change; it does not
estimate success rates:

| Configuration | Accepted | Converged and accepted | False claims |
| --- | --- | --- | --- |
| Clarabel, explicit | 37 | 34 | 0 |
| active-set, explicit | 41 | 38 | 0 |
| `auto` (active-set with elastic mode) | 41 | 40 | 0 |

Explicit Clarabel and active-set records equal those of 0.18.0. Against plain
active-set, `auto` gains eight problems and loses eight. In elastic mode,
problems without a row to relax now run the plain QP (see
[elastic mode](elastic-mode.md#the-driver)); before, `auto` lost three
unconstrained problems there and accepted 38. A stress campaign of 3,000
automatic runs on the random and complementarity families made no false
claims.

Elastic mode can change the iteration path, callback counts and failure
status. An inconsistent problem may now finish `locally_infeasible` instead
of exhausting restoration. That status is a local result, never proof that
the nonlinear problem has no feasible point. The historical elastic variant
also lost some problems; the new policy does not guarantee improvement on
every model. Pin both backend and elastic setting to reproduce old runs.

Named benchmark variants pin their settings explicitly. The separate `auto`
variant tests the public default, and the report includes it in false-claim
checks. Historical benchmark tables and logs retain their original meaning.
