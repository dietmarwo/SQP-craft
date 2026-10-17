"""The solvers of the harness, each run the same way on one `S2mpjProblem`.

`run(name, problem)` solves from the problem's (clipped) start within the
common budgets and returns one record:
- `status`: the solver's own status, as a string;
- `claimed`: whether the solver claims a first-order point;
- `x`, `constraint_multipliers`, `variable_multipliers` (the latter two in
  sqp-craft's convention, ``∇f = Jᵀλ + z``, or `None`);
- `iterations`, `functions`, `derivatives` (distinct points, counted by the
  adapter), `hessians` (exact-Hessian runs only) and `seconds`.

Budgets: 500 iterations and 60 s for every solver; sqp-craft also has
`max_callback_calls = 5000`.

Every sqp-craft variant is a set of documented options on top of the
defaults; nothing is tuned per problem. A model that raises or returns a
nonfinite value is reported to sqp-craft as undefined (`UndefinedPointError`,
the documented contract), to Uno as NaN (its evaluation-error path) and to
SLSQP as NaN.
"""
from __future__ import annotations

import math
import time

import numpy as np

MAX_ITERATIONS = 500
TIME_LIMIT = 60.0
MAX_CALLBACKS = 5000

# sqp-craft variants: name -> (SqpOptions fields, declare the linear rows).
ACTIVE = dict(qp_solver="active-set")
SQP_VARIANTS: dict[str, tuple[dict, bool]] = {
    "clarabel": ({}, False),  # the current default
    "active-set": (ACTIVE, False),
    "active-set+soc": ({**ACTIVE, "second_order_correction": True}, False),
    "active-set+al": ({**ACTIVE, "merit": "augmented-lagrangian"}, False),
    "active-set+augmented": ({**ACTIVE, "hessian_update": "augmented"}, False),
    "active-set+scale-jacobian": ({**ACTIVE, "scaling": "jacobian"}, False),
    "active-set+linear": (ACTIVE, True),
    "active-set+linear+scale-linear": ({**ACTIVE, "scaling": "linear"}, True),
    "active-set+elastic": ({**ACTIVE, "elastic_mode": True}, False),
    # Everything SNOPT does that sqp-craft has as an option.
    "active-set+snopt": ({**ACTIVE, "merit": "augmented-lagrangian",
                          "hessian_update": "augmented", "scaling": "linear",
                          "elastic_mode": True, "violation_limit": 10.0}, True),
    "clarabel+al": ({"merit": "augmented-lagrangian"}, False),
}
# Experiments: candidate rules not (yet) in sqp-craft, emulated with options.
# "b0-gradient" starts BFGS from max(1, ‖∇f(x₀)‖∞)·I instead of I, so the
# first step is at most of unit length (a stand-in for a first-step scaling or
# step limit; resets use the same matrix).
EXPERIMENTS: dict[str, tuple[dict, bool]] = {
    "active-set+b0-gradient": (ACTIVE, False),
}
REPORT_FIELDS = ["qp_solves", "qp_iterations", "rejected_trials", "undefined_trials",
                 "hessian_resets", "bfgs_damped_updates", "bfgs_modified_updates",
                 "elastic_steps", "second_order_steps", "violation_limit_rejections",
                 "linear_projection", "restoration_attempts", "restoration_steps",
                 "qp_seconds", "callback_seconds", "qp_status"]
COMPARATORS = ["slsqp", "uno-filtersqp", "uno-ipopt", "uno-filtersqp-exact"]
SOLVERS = list(SQP_VARIANTS) + COMPARATORS + list(EXPERIMENTS)


def warm_up() -> None:
    """Imports every solver's modules, so no solve is charged for them."""
    import scipy.optimize  # noqa: F401
    import sqp_craft  # noqa: F401

    try:
        import unopy  # noqa: F401
    except ImportError:
        pass


def available(name: str) -> bool:
    if name.startswith("uno"):
        try:
            import unopy  # noqa: F401
        except ImportError:
            return False
    return True


def run(name: str, problem) -> dict:
    problem.reset_counts()
    began = time.perf_counter()
    if name in SQP_VARIANTS or name in EXPERIMENTS:
        record = _sqp_craft(name, problem)
    elif name == "slsqp":
        record = _slsqp(problem)
    elif name.startswith("uno-"):
        preset = "filtersqp" if name.startswith("uno-filtersqp") else "ipopt"
        record = _uno(problem, preset, exact=name.endswith("-exact"))
    else:
        raise ValueError(f"unknown solver {name!r}")
    record["seconds"] = time.perf_counter() - began
    record["functions"], record["derivatives"] = problem.counts
    record.setdefault("hessians", 0)
    return record


def _finite_list(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    return [float(v) for v in a] if np.all(np.isfinite(a)) else None


# --- sqp-craft ----------------------------------------------------------------
class _SqpModel:
    def __init__(self, problem):
        self.problem = problem

    def functions(self, x):
        from sqp_craft import UndefinedPointError

        try:
            f, c = self.problem.functions(x)
        except (ArithmeticError, ValueError) as error:
            raise UndefinedPointError(f"{type(error).__name__}: {error}") from None
        if not (math.isfinite(f) and np.all(np.isfinite(c))):
            raise UndefinedPointError("nonfinite function values")
        return f, c.tolist()

    def derivatives(self, x):
        from sqp_craft import UndefinedPointError

        try:
            g, j = self.problem.derivatives(x)
        except (ArithmeticError, ValueError) as error:
            raise UndefinedPointError(f"{type(error).__name__}: {error}") from None
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(j))):
            raise UndefinedPointError("nonfinite derivatives")
        return g.tolist(), j.tolist()


def _sqp_craft(name: str, problem) -> dict:
    from sqp_craft import SqpOptions, solve

    fields, linear = SQP_VARIANTS.get(name) or EXPERIMENTS[name]
    if name.endswith("+b0-gradient"):
        g = problem.gradient(problem.x0)
        fields = {**fields, "initial_hessian_scale": max(1.0, float(np.max(np.abs(g)))
                                                         if g.size else 1.0)}
    options = SqpOptions(max_iterations=MAX_ITERATIONS, max_callback_calls=MAX_CALLBACKS,
                         time_limit_seconds=TIME_LIMIT, **fields)
    try:
        r = solve(problem.spec(linear=linear), _SqpModel(problem), list(problem.x0),
                  method="sqp", method_options=options)
    except Exception as error:  # an error, not a status: recorded as such
        return dict(status="error", claimed=False, x=None, constraint_multipliers=None,
                    variable_multipliers=None, iterations=None,
                    message=f"{type(error).__name__}: {error}"[:300])
    m = r.multipliers
    return dict(status=r.status, claimed=r.status == "converged", x=r.candidate,
                constraint_multipliers=list(m.constraints) if m else None,
                variable_multipliers=list(m.variables) if m else None,
                iterations=r.iterations, message=r.message[:300],
                report={key: getattr(r.report, key) for key in REPORT_FIELDS})


# --- SciPy SLSQP ----------------------------------------------------------------
def _slsqp(problem) -> dict:
    from scipy.optimize import Bounds, minimize

    n, m = problem.n, problem.m
    cache: dict = {}

    def values(x):
        key = x.tobytes()
        if cache.get("f") is None or cache["f"][0] != key:
            try:
                f, c = problem.functions(x)
            except (ArithmeticError, ValueError):
                f, c = math.nan, np.full(m, math.nan)
            cache["f"] = (key, f, c)
        return cache["f"][1], cache["f"][2]

    def derivatives(x):
        key = x.tobytes()
        if cache.get("d") is None or cache["d"][0] != key:
            try:
                g, j = problem.derivatives(x)
                J = problem.dense_jacobian(j)
            except (ArithmeticError, ValueError):
                g, J = np.full(n, math.nan), np.full((m, n), math.nan)
            cache["d"] = (key, g, J)
        return cache["d"][1], cache["d"][2]

    eq = np.flatnonzero(problem.clower == problem.cupper)
    lo = np.flatnonzero(np.isfinite(problem.clower) & (problem.clower != problem.cupper))
    up = np.flatnonzero(np.isfinite(problem.cupper) & (problem.clower != problem.cupper))
    constraints = []
    if eq.size:
        constraints.append(dict(type="eq", fun=lambda x: values(x)[1][eq] - problem.clower[eq],
                                jac=lambda x: derivatives(x)[1][eq]))
    if lo.size or up.size:
        constraints.append(dict(
            type="ineq",
            fun=lambda x: np.concatenate([values(x)[1][lo] - problem.clower[lo],
                                          problem.cupper[up] - values(x)[1][up]]),
            jac=lambda x: np.vstack([derivatives(x)[1][lo], -derivatives(x)[1][up]])))
    bounds = Bounds(problem.lower, problem.upper) if (
        np.any(np.isfinite(problem.lower)) or np.any(np.isfinite(problem.upper))) else None
    try:
        r = minimize(lambda x: values(x)[0], problem.x0.copy(), jac=lambda x: derivatives(x)[0],
                     method="SLSQP", bounds=bounds, constraints=constraints,
                     options=dict(maxiter=MAX_ITERATIONS, ftol=1e-12))
    except Exception as error:
        return dict(status="error", claimed=False, x=None, constraint_multipliers=None,
                    variable_multipliers=None, iterations=None,
                    message=f"{type(error).__name__}: {error}"[:300])
    return dict(status=f"slsqp-{r.status}", claimed=bool(r.success), x=_finite_list(r.x),
                constraint_multipliers=None, variable_multipliers=None,
                iterations=int(r.nit), message=str(r.message)[:300])


# --- Uno -------------------------------------------------------------------------
def _uno(problem, preset: str, exact: bool) -> dict:
    import unopy

    n, m = problem.n, problem.m
    nan = math.nan

    def guarded(compute, size):
        def call(x, out=None):
            x = np.asarray(x, dtype=float)
            try:
                value = compute(x)
            except (ArithmeticError, ValueError):
                value = nan if out is None else np.full(size, nan)
            if out is None:
                return float(value)
            out[:] = value
        return call

    def objective(x):
        problem.function_points.add(tuple(x))
        return problem.objective(x)

    def gradient(x):
        problem.derivative_points.add(tuple(x))
        return problem.gradient(x)

    def constraints(x):
        problem.function_points.add(tuple(x))
        return problem.constraints(x)

    def jacobian(x):
        problem.derivative_points.add(tuple(x))
        return problem.jacobian_values(x)

    model = unopy.Model(unopy.PROBLEM_NONLINEAR, n, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds(list(problem.lower))
    model.set_variables_upper_bounds(list(problem.upper))
    model.set_objective(unopy.MINIMIZE, guarded(objective, 0), guarded(gradient, n))
    if m:
        model.set_constraints(m, guarded(constraints, m), list(problem.clower),
                              list(problem.cupper), problem.nnz,
                              [int(i) for i in problem._rows], [int(j) for j in problem._cols],
                              guarded(jacobian, problem.nnz))
    elif exact:
        # unopy 0.4.15 fails to pass an empty multiplier vector to the Hessian
        # callback ("PyCapsule_New called with null pointer"); one free row
        # with a zero Jacobian entry avoids it without changing the problem.
        def free(x, out):
            out[:] = 0.0

        model.set_constraints(1, free, [-math.inf], [math.inf], 1, [0], [0], free)
    hessians = [0]
    if exact:
        rows, cols = np.tril_indices(n)

        def hessian(x, sigma, multipliers, out):
            hessians[0] += 1
            x = np.asarray(x, dtype=float)
            try:
                y = np.asarray(multipliers, float)[:m]
                H = _lagrangian_hessian(problem, x, float(sigma), y)
                out[:] = H[rows, cols]
            except (ArithmeticError, ValueError):
                out[:] = np.full(rows.size, nan)

        model.set_lagrangian_hessian(int(rows.size), unopy.LOWER_TRIANGLE,
                                     [int(i) for i in rows], [int(j) for j in cols], hessian)
        model.set_lagrangian_sign_convention(unopy.MULTIPLIER_NEGATIVE)
    model.set_initial_primal_iterate(list(problem.x0))
    solver = unopy.UnoSolver()
    solver.set_preset(preset)
    solver.set_option("logger", "SILENT")
    solver.set_option("max_iterations", MAX_ITERATIONS)
    solver.set_option("time_limit", TIME_LIMIT)
    if not exact:
        solver.set_option("hessian_model", "LBFGS")
    try:
        r = solver.optimize(model)
    except Exception as error:
        return dict(status="error", claimed=False, x=None, constraint_multipliers=None,
                    variable_multipliers=None, iterations=None, hessians=hessians[0],
                    message=f"{type(error).__name__}: {error}"[:300])
    optimization = str(r.optimization_status).split(".")[-1]
    solution = str(r.solution_status).split(".")[-1]
    claimed = optimization == "SUCCESS" and solution == "FEASIBLE_KKT_POINT"
    z = None
    if r.lower_bound_dual_solution is not None and r.upper_bound_dual_solution is not None:
        z = _finite_list(np.asarray(r.lower_bound_dual_solution, float)
                         + np.asarray(r.upper_bound_dual_solution, float))
    lam = _finite_list(np.asarray(r.constraint_dual_solution, float)[:m]) if m else []
    return dict(status=f"{optimization}/{solution}", claimed=claimed,
                x=_finite_list(r.primal_solution), constraint_multipliers=lam,
                variable_multipliers=z, iterations=int(r.number_iterations),
                hessians=hessians[0], message="")


def _lagrangian_hessian(problem, x, sigma, multipliers) -> np.ndarray:
    """σ∇²f − Σλᵢ∇²cᵢ from S2MPJ, whose Lagrangian is f + yᵀc."""
    raw, column = problem.raw, x.reshape(-1, 1)

    def dense(H):
        return H.toarray() if hasattr(H, "toarray") else np.asarray(H, dtype=float)

    objective = dense(raw.fgHx(column)[2]) if problem.has_objective else np.zeros((problem.n,) * 2)
    if not problem.m:
        return sigma * objective
    if problem.has_objective:
        # ∇²(f − λᵀc) from one call, then the objective's weight corrected.
        lagrangian = dense(raw.LgHxy(column, -multipliers.reshape(-1, 1))[2])
        return lagrangian + (sigma - 1.0) * objective
    rows = raw.cJHx(column)[2]
    return -sum(l * dense(H) for l, H in zip(multipliers, rows) if l != 0.0) + 0.0 * objective
