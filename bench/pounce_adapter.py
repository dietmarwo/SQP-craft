"""Black-box POUNCE 0.12.0 comparator through the public Python Problem API."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import warnings

import numpy as np

from .callbacks import CallbackWork, lagrangian_hessian

VERSION = "0.12.0"
COMMIT = "a31dab9c21a25ce0e5e8a93ab78f0250c9f0f0fd"
MODES = {
    "pounce-sqp-exact": "exact",
    "pounce-sqp-lbfgs": "lbfgs",
    "pounce-sqp-damped": "damped-bfgs",
}


def require_wheel():
    try:
        installed = version("pounce-solver")
    except PackageNotFoundError:
        raise ImportError(f"install optional pounce-solver=={VERSION}") from None
    if installed != VERSION:
        raise ImportError(f"expected pounce-solver=={VERSION}, found {installed}")
    import pounce

    return pounce


class FirstDerivatives:
    def __init__(self, problem):
        self.problem = problem
        self.work = CallbackWork()
        self._cache = {}

    def _read(self, request, group, x, index):
        self.work.requests[request] += 1
        key = x.tobytes()
        hit = self._cache.get(group)
        if hit is None or hit[0] != key:
            values = self.work.evaluate(group, getattr(self.problem, group), x)
            self._cache[group] = key, values
        else:
            values = hit[1]
        return values[index]

    def objective(self, x):
        return self._read("objective", "functions", x, 0)

    def constraints(self, x):
        return self._read("constraints", "functions", x, 1)

    def gradient(self, x):
        return self._read("gradient", "derivatives", x, 0)

    def jacobian(self, x):
        return self._read("jacobian", "derivatives", x, 1)

    def jacobianstructure(self):
        return self.problem._rows, self.problem._cols


class Exact(FirstDerivatives):
    def __init__(self, problem):
        super().__init__(problem)
        self.triangle = np.tril_indices(problem.n)

    def hessianstructure(self):
        return self.triangle

    def hessian(self, x, lagrange, obj_factor):
        self.work.requests["hessian"] += 1
        matrix = self.work.evaluate(
            "hessians", lagrangian_hessian, self.problem, x, obj_factor, -lagrange,
            self.work,
        )
        return matrix[self.triangle]


def _vector(info, key, size):
    values = np.asarray(info.get(key), dtype=float)
    return values if values.shape == (size,) and np.isfinite(values).all() else None


def duals(info, n, m):
    raw = {key: _vector(info, key, size)
           for key, size in (("mult_g", m), ("mult_x_L", n), ("mult_x_U", n))}
    lam, lower, upper = raw["mult_g"], raw["mult_x_L"], raw["mult_x_U"]
    return (None if lam is None else (-lam).tolist(),
            None if lower is None or upper is None else (lower - upper).tolist(),
            {key: None if value is None else value.tolist() for key, value in raw.items()})


def options(mode, max_iterations):
    fields = dict(algorithm="active-set-sqp", sqp_hessian=mode, sqp_globalization="filter",
                  linear_solver="feral", nlp_scaling_method="none", bound_relax_factor=0.0,
                  nlp_lower_bound_inf=-1e20, nlp_upper_bound_inf=1e20,
                  sqp_max_iter=max_iterations, sqp_tol=1e-8, sqp_constr_viol_tol=1e-6,
                  sqp_dual_inf_tol=1e-4, print_level=0)
    if mode == "lbfgs":
        fields["sqp_lbfgs_max_history"] = 6
    return fields


def status_fields(info):
    code = int(info["status"])
    return dict(status=info["status_msg"], status_code=code, claimed=code == 0,
                acceptable=code == 1)


def solve(problem, mode, max_iterations):
    pounce = require_wheel()
    model = Exact(problem) if mode == "exact" else FirstDerivatives(problem)
    fields = options(mode, max_iterations)
    record = dict(options=fields, x=None, constraint_multipliers=None, variable_multipliers=None,
                  iterations=None, reported_objective=None, acceptable=False)
    try:
        if mode == "exact" and not all(callable(getattr(model, method, None))
                                       for method in ("hessian", "hessianstructure")):
            raise ValueError("exact Hessian was requested without both callbacks")
        solver = pounce.Problem(n=problem.n, m=problem.m, problem_obj=model,
                                lb=problem.lower, ub=problem.upper,
                                cl=problem.clower, cu=problem.cupper)
        for key, value in fields.items():
            solver.add_option(key, value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x, info = solver.solve(x0=problem.x0.copy())
        record["warnings"] = [str(item.message) for item in caught]
        if any("exact hessian was requested" in message.lower() for message in record["warnings"]):
            raise RuntimeError("the binding downgraded the requested exact Hessian")
    except BaseException as error:
        # PyO3 may turn a Rust panic during a failed TNLP evaluation into a
        # BaseException. Keep callback counters for that case.
        if not isinstance(error, Exception) and type(error).__name__ != "PanicException":
            raise
        record.update(status="evaluation_error" if any(model.work.failures.values()) else "error",
                      claimed=False, message=f"{type(error).__name__}: {error}"[:300])
    else:
        point = np.asarray(x, dtype=float)
        lam, z, raw = duals(info, problem.n, problem.m)
        objective = float(info["obj_val"])
        record.update(status_fields(info))
        record.update(
            x=point.tolist() if point.shape == (problem.n,) and np.isfinite(point).all() else None,
            constraint_multipliers=lam, variable_multipliers=z, raw_multipliers=raw,
            iterations=int(info["iter_count"]),
            reported_objective=objective if np.isfinite(objective) else None,
            message="",
            report=dict(qp_solver=fields["linear_solver"], qp_solves=info.get("n_qp_solves"),
                        qp_working_set_changes=info.get("n_qp_ws_changes"), qp_iterations=None,
                        restoration_calls=info.get("restoration_calls"),
                        quality_escalations=info.get("quality_escalations")),
        )
    record["hessians"] = model.work.evaluations["hessians"]
    record["callback_work"] = model.work.record()
    return record
