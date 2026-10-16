"""Compare sqp-craft with SciPy's SLSQP on the same models and starts.

    pip install '.[bench]'
    python examples/compare_baseline.py            # JSON on stdout
    python examples/compare_baseline.py --table    # Markdown table

Acceptance is decided by one independent evaluator for both solvers, written
here in NumPy (it does not call sqp-craft's Rust KKT code). At the returned
point it recomputes objective, constraints and derivatives from the model and
checks primal feasibility in original units, the objective against a
reference, and a multiplier certificate: stationarity, sign and
complementarity of either the solver's own complete multipliers (sqp-craft)
or multipliers reconstructed by least squares over the active bounds (SciPy
returns no bound multipliers). Any certificate that passes proves first-order
stationarity; the report says which one was used. Neither solver's own status
is trusted.
Model calls are counted at the model; the SLSQP adapter caches by point so
its separate objective/constraint/Jacobian requests are not double counted.
Single runs of small problems: timings carry no performance conclusion.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import orbit_transfer  # noqa: E402

from sqp_craft import ProblemSpec, SqpOptions, solve  # noqa: E402

INF = math.inf
PRIMAL, STATIONARITY, COMPLEMENTARITY, SIGN = 1e-7, 1e-6, 1e-6, 1e-8


class Counted:
    """Counts real model calls; repeated requests at the same point are cached."""

    def __init__(self, model):
        self.model = model
        self.functions_calls = 0
        self.derivatives_calls = 0
        self._f = self._d = None

    def functions(self, x):
        key = tuple(float(v) for v in x)
        if self._f is None or self._f[0] != key:
            self.functions_calls += 1
            f, c = self.model.functions(list(key))
            self._f = (key, (float(f), [float(v) for v in c]))
            self._d = None
        return self._f[1]

    def derivatives(self, x):
        key = tuple(float(v) for v in x)
        if self._d is None or self._d[0] != key:
            self.functions(key)  # keep the grouped protocol: functions first
            self.derivatives_calls += 1
            g, j = self.model.derivatives(list(key))
            self._d = (key, ([float(v) for v in g], [float(v) for v in j]))
        return self._d[1]


@dataclass
class Case:
    name: str
    spec: ProblemSpec
    make_model: object
    start: list
    reference: float
    reference_tolerance: float
    max_iterations: int = 100


def dense(lower, upper, rlo=(), rup=()):
    n, m = len(lower), len(rlo)
    return ProblemSpec(lower=lower, upper=upper, constraint_lower=rlo, constraint_upper=rup,
                       jacobian_indptr=[i * n for i in range(m + 1)],
                       jacobian_indices=list(range(n)) * m)


class Quadratic:
    def __init__(self, row=False):
        self.row = row

    def functions(self, x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, [x[0] + x[1]] if self.row else []

    def derivatives(self, x):
        return [2 * (x[0] - 1), 2 * (x[1] - 2)], [1.0, 1.0] if self.row else []


class Circle:
    def functions(self, x):
        return 0.5 * ((x[0] - 2) ** 2 + x[1] ** 2), [x[0] ** 2 + x[1] ** 2]

    def derivatives(self, x):
        return [x[0] - 2, x[1]], [2 * x[0], 2 * x[1]]


class Bound:
    def functions(self, x):
        return (x[0] + 1) ** 2, []

    def derivatives(self, x):
        return [2 * (x[0] + 1)], []


class Rosenbrock:
    def functions(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, []

    def derivatives(self, x):
        return [-400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1), 200 * (x[1] - x[0] ** 2)], []


class Hs071:
    def functions(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2], [math.prod(x), sum(v * v for v in x)]

    def derivatives(self, x):
        p = math.prod(x)
        g = [x[3] * (2 * x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1, x[0] * (x[0] + x[1] + x[2])]
        return g, [p / v for v in x] + [2 * v for v in x]


def cases():
    free2 = dense([-INF] * 2, [INF] * 2)
    yield Case("quadratic", free2, Quadratic, [0.0, 0.0], 0.0, 1e-8)
    yield Case("quadratic", free2, Quadratic, [-2.0, 4.0], 0.0, 1e-8)
    equality = dense([-INF] * 2, [INF] * 2, [1.0], [1.0])
    yield Case("equality", equality, lambda: Quadratic(True), [0.0, 0.0], 2.0, 1e-6)
    yield Case("equality", equality, lambda: Quadratic(True), [4.0, -3.0], 2.0, 1e-6)
    circle = dense([-INF] * 2, [INF] * 2, [-INF], [1.0])
    yield Case("circle", circle, Circle, [0.4, 0.2], 0.5, 1e-6)
    yield Case("circle", circle, Circle, [0.2, 0.4], 0.5, 1e-6)
    yield Case("bound", dense([0.0], [INF]), Bound, [1.0], 1.0, 1e-6)
    yield Case("rosenbrock", free2, Rosenbrock, [-1.2, 1.0], 0.0, 1e-8)
    hs071 = dense([1.0] * 4, [5.0] * 4, [25.0, 40.0], [INF, 40.0])
    for start in ([1.0, 5.0, 5.0, 1.0], [2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]):
        yield Case("hs071", hs071, Hs071, start, 17.0140173, 1e-6)
    for n_intervals, reference in ((20, -1.52206), (40, -1.52449)):
        spec, model, x0 = orbit_transfer.build(n_intervals)
        yield Case(f"orbit-{n_intervals}", spec, lambda m=model: m, list(x0), reference, 1e-5, 300)


def evaluate(case: Case, x, multipliers=None) -> dict:
    """Independent first-order check at x; see the module docstring."""
    spec, model = case.spec, case.make_model()
    x = np.asarray(x, dtype=float)
    f, c = model.functions(list(x))
    g, values = model.derivatives(list(x))
    g, c = np.asarray(g, float), np.asarray(c, float)
    n, m = spec.n, spec.m
    jac = np.zeros((m, n))
    indptr, indices = spec.jacobian_indptr, spec.jacobian_indices
    for i in range(m):
        for k in range(indptr[i], indptr[i + 1]):
            jac[i, indices[k]] = values[k]
    # One block of [rows; variables] for values, bounds and gradients.
    value = np.concatenate([c, x])
    lower = np.concatenate([spec.constraint_lower, spec.lower])
    upper = np.concatenate([spec.constraint_upper, spec.upper])
    gradients = np.vstack([jac, np.eye(n)])
    primal = float(max(0.0, np.max(lower - value), np.max(value - upper)))

    def certificate(mu):
        stationarity = float(np.max(np.abs(g - gradients.T @ mu))) if n else 0.0
        sign = complementarity = 0.0
        for v, lo, up, u in zip(value, lower, upper, mu):
            if u > 0:  # selects the lower bound
                sign = max(sign, u if not math.isfinite(lo) else 0.0)
                complementarity = max(complementarity, u * abs(v - lo) if math.isfinite(lo) else 0.0)
            elif u < 0:
                sign = max(sign, -u if not math.isfinite(up) else 0.0)
                complementarity = max(complementarity, -u * abs(up - v) if math.isfinite(up) else 0.0)
        return dict(stationarity=stationarity, complementarity=float(complementarity),
                    dual_sign=float(sign))

    def passes(cert):
        return (cert["stationarity"] <= STATIONARITY and cert["complementarity"] <= COMPLEMENTARITY
                and cert["dual_sign"] <= SIGN)

    def near(v, bound):
        return math.isfinite(bound) and abs(v - bound) <= 1e-6 * (1 + abs(bound))

    # Least-squares multipliers on the active set.
    active, kinds = [], []
    for i, (v, lo, up) in enumerate(zip(value, lower, upper)):
        if lo == up:
            active.append(i); kinds.append(0)
        elif near(v, lo):
            active.append(i); kinds.append(1)
        elif near(v, up):
            active.append(i); kinds.append(-1)
    mu = np.zeros(m + n)
    if active:
        mu[active] = np.linalg.lstsq(gradients[active].T, g, rcond=None)[0]
    # certificate() rejects wrong signs: a negative multiplier at an active lower
    # bound selects the upper side, failing complementarity or the sign test.
    least_squares = certificate(mu)
    chosen, source = least_squares, "least-squares"
    if multipliers is not None:
        own = certificate(np.concatenate([multipliers.constraints, multipliers.variables]))
        if passes(own) or not passes(least_squares):
            chosen, source = own, "solver"
    objective_error = abs(f - case.reference)
    accepted = (primal <= PRIMAL and passes(chosen)
                and objective_error <= case.reference_tolerance * max(1.0, abs(case.reference)))
    return dict(objective=float(f), primal=primal, **chosen, certificate=source,
                objective_error=float(objective_error), accepted=bool(accepted))


def run_sqp_craft(case: Case) -> dict:
    model = Counted(case.make_model())
    began = perf_counter()
    r = solve(case.spec, model, case.start, method="sqp",
              method_options=SqpOptions(max_iterations=case.max_iterations,
                                        max_callback_calls=20 * case.max_iterations + 100))
    seconds = perf_counter() - began
    check = evaluate(case, r.candidate, r.multipliers) if r.candidate is not None else None
    return dict(solver="sqp-craft", status=r.status, iterations=r.iterations, seconds=seconds,
                functions=model.functions_calls, derivatives=model.derivatives_calls, check=check)


def run_slsqp(case: Case) -> dict:
    spec, model = case.spec, Counted(case.make_model())
    n, m = spec.n, spec.m
    indptr, indices = spec.jacobian_indptr, spec.jacobian_indices

    def jacobian(x):
        dense_jac = np.zeros((m, n))
        values = model.derivatives(x)[1]
        for i in range(m):
            for k in range(indptr[i], indptr[i + 1]):
                dense_jac[i, indices[k]] = values[k]
        return dense_jac

    constraints = []
    for i, (lo, up) in enumerate(zip(spec.constraint_lower, spec.constraint_upper)):
        if lo == up:
            constraints.append(dict(type="eq", fun=lambda x, i=i, b=lo: model.functions(x)[1][i] - b,
                                    jac=lambda x, i=i: jacobian(x)[i]))
            continue
        if math.isfinite(lo):
            constraints.append(dict(type="ineq", fun=lambda x, i=i, b=lo: model.functions(x)[1][i] - b,
                                    jac=lambda x, i=i: jacobian(x)[i]))
        if math.isfinite(up):
            constraints.append(dict(type="ineq", fun=lambda x, i=i, b=up: b - model.functions(x)[1][i],
                                    jac=lambda x, i=i: -jacobian(x)[i]))
    bounds = [(None if not math.isfinite(lo) else lo, None if not math.isfinite(up) else up)
              for lo, up in zip(spec.lower, spec.upper)]
    began = perf_counter()
    res = minimize(lambda x: model.functions(x)[0], np.asarray(case.start, float),
                   jac=lambda x: np.asarray(model.derivatives(x)[0]), bounds=bounds,
                   constraints=constraints, method="SLSQP",
                   options=dict(ftol=1e-12, maxiter=case.max_iterations))
    seconds = perf_counter() - began
    return dict(solver="scipy-SLSQP", status=str(res.message), iterations=int(res.nit),
                seconds=seconds, functions=model.functions_calls,
                derivatives=model.derivatives_calls, check=evaluate(case, res.x))


def run() -> dict:
    runs = []
    for case in cases():
        for runner in (run_sqp_craft, run_slsqp):
            record = runner(case)
            record.update(case=case.name, start=[float(v) for v in case.start][:4])
            runs.append(record)
    return dict(scipy=scipy.__version__, numpy=np.__version__,
                thresholds=dict(primal=PRIMAL, stationarity=STATIONARITY,
                                complementarity=COMPLEMENTARITY, dual_sign=SIGN),
                runs=runs)


def table(data: dict) -> str:
    lines = ["| Case | Start | Solver | Status | Accepted | Iterations | f / ∇f calls |",
             "| --- | --- | --- | --- | :---: | ---: | ---: |"]
    for r in data["runs"]:
        start = "x0 (N+1 nodes)" if r["case"].startswith("orbit") else str(r["start"])
        accepted = "yes" if r["check"] and r["check"]["accepted"] else "**no**"
        lines.append(f"| {r['case']} | {start} | {r['solver']} | {r['status']} | {accepted} | "
                     f"{r['iterations']} | {r['functions']} / {r['derivatives']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    data = run()
    print(table(data) if "--table" in sys.argv else json.dumps(data, indent=2, allow_nan=False))
