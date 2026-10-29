"""Direct native-SQP tutorial runner with independent KKT reconstruction."""

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import numpy as np
from sqp_craft import ProblemSpec, SqpOptions, Options, solve, check_derivatives


def plain(v):
    if is_dataclass(v):
        return plain(asdict(v))
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


class Case:
    def __init__(
        self, lower, upper, cl, cu, fun, deriv, start, linear_rows=None, pattern=None
    ):
        self.lower = np.array(lower, float)
        self.upper = np.array(upper, float)
        self.cl = np.array(cl, float)
        self.cu = np.array(cu, float)
        self.fun, self.deriv, self.start = fun, deriv, np.array(start, float)
        n = len(lower)
        m = len(cl)
        self.pattern = (
            [list(range(n)) for _ in range(m)] if pattern is None else pattern
        )
        ptr = np.cumsum([0] + [len(r) for r in self.pattern])
        ix = [j for row in self.pattern for j in row]
        self.spec = ProblemSpec(
            lower=lower,
            upper=upper,
            constraint_lower=cl,
            constraint_upper=cu,
            jacobian_indptr=ptr,
            jacobian_indices=ix,
            linear_rows=linear_rows,
        )

    def functions(self, x):
        return self.fun(np.asarray(x))

    def derivatives(self, x):
        g, J = self.deriv(np.asarray(x))
        J = np.asarray(J)
        return np.asarray(g).tolist(), [
            J[i, j] for i, row in enumerate(self.pattern) for j in row
        ]

    def check(self, x, multipliers=None):
        x = np.asarray(x)
        if x.shape != self.lower.shape or not np.all(np.isfinite(x)):
            raise ValueError("invalid candidate")
        f, c = self.fun(x)
        c = np.asarray(c)
        if c.shape != self.cl.shape or not np.isfinite(f) or not np.all(np.isfinite(c)):
            raise ValueError("invalid function evaluation")
        primal = max(
            [0.0]
            + list(self.lower - x)
            + list(x - self.upper)
            + list(self.cl - c)
            + list(c - self.cu)
        )
        result = dict(
            objective=float(f), primal=float(primal), feasible=bool(primal <= 1e-7)
        )
        if multipliers is not None:
            lam = np.asarray(multipliers.constraints)
            z = np.asarray(multipliers.variables)
            g, J = self.deriv(x)
            if not np.all(np.isfinite(g)) or not np.all(np.isfinite(J)):
                raise ValueError("invalid derivative evaluation")
            assert np.all(np.isfinite(lam)) and np.all(np.isfinite(z))
            stationarity = np.max(np.abs(g - np.asarray(J).T @ lam - z))
            comp = 0.0
            sign = 0.0
            for val, l, u, dual in zip(
                np.r_[x, c],
                np.r_[self.lower, self.cl],
                np.r_[self.upper, self.cu],
                np.r_[z, lam],
            ):
                bound = l if dual > 0 else u
                if not np.isfinite(bound):
                    sign = max(sign, abs(dual))
                else:
                    comp = max(comp, abs(dual * (val - bound)))
            result.update(
                stationarity=float(stationarity),
                complementarity=float(comp),
                dual_sign=float(sign),
            )
            result["first_order"] = bool(
                result["feasible"]
                and stationarity <= 1e-6
                and comp <= 1e-6
                and sign <= 1e-8
            )
        return result

    def run(
        self,
        backend="active-set",
        scaling="none",
        start=None,
        iterations=500,
        check=True,
        seconds=60,
    ):
        x = self.start if start is None else np.asarray(start)
        derivative_check = check_derivatives(self.spec, self, x) if check else None
        if derivative_check is not None:
            assert derivative_check.passed, plain(derivative_check)
        r = solve(
            self.spec,
            self,
            x,
            method="sqp",
            method_options=SqpOptions(
                qp_solver=backend,
                scaling=scaling,
                max_iterations=iterations,
                max_callback_calls=20000,
                time_limit_seconds=seconds,
                record_history=True,
            ),
            options=Options(threads=1),
        )
        audit = (
            self.check(r.candidate, r.multipliers) if r.candidate is not None else None
        )
        if r.status == "converged":
            assert audit and audit.get("first_order"), (r.status, audit)
        return r, dict(
            native=plain(r),
            independent=audit,
            derivative_check=plain(derivative_check),
            backend=backend,
            scaling=scaling,
            claim="first_order"
            if r.status == "converged" and audit.get("first_order")
            else "feasible"
            if audit and audit["feasible"]
            else "none",
        )


def complex_jac(fun, x):
    """Complex-step derivative for explicitly holomorphic teaching algebra."""
    x = np.asarray(x, float)
    base = np.atleast_1d(fun(x))
    J = np.empty((len(base), len(x)))
    for j in range(len(x)):
        z = x.astype(complex)
        z[j] += 1e-25j
        J[:, j] = np.imag(fun(z)) / 1e-25
    return J
