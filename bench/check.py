"""The independent first-order check applied to every run, whatever its solver
claims.

At the returned point it recomputes, in NumPy from the problem's own values:
- the primal violation of rows and bounds;
- the stationarity, complementarity and dual-sign residuals of
  ``∇f = Jᵀλ + z``, where a positive multiplier belongs to an active lower
  bound (sqp-craft's convention).

It uses the solver's multipliers when they pass. Otherwise it tries
least-squares multipliers on the active set, so a different sign convention
cannot fail a correct point.
"""
from __future__ import annotations

import math

import numpy as np

PRIMAL = 1e-6
DUAL = 1e-6  # times max(1, ‖∇f‖∞)


def _certificate(g, gradients, value, lower, upper, mu):
    stationarity = float(np.max(np.abs(g - gradients.T @ mu))) if g.size else 0.0
    sign = complementarity = 0.0
    for v, lo, up, u in zip(value, lower, upper, mu):
        if u > 0:
            if math.isfinite(lo):
                complementarity = max(complementarity, u * abs(v - lo))
            else:
                sign = max(sign, u)
        elif u < 0:
            if math.isfinite(up):
                complementarity = max(complementarity, -u * abs(up - v))
            else:
                sign = max(sign, -u)
    return dict(stationarity=stationarity, complementarity=float(complementarity),
                dual_sign=float(sign))


def check(problem, x, constraint_multipliers=None, variable_multipliers=None) -> dict:
    """Residuals at `x` and whether they pass (`accepted`)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != problem.n or not np.all(np.isfinite(x)):
        return dict(accepted=False, reason="no finite point")
    try:
        f = problem.objective(x)
        c = problem.constraints(x)
        g = problem.gradient(x)
        J = (problem.dense_jacobian(problem.jacobian_values(x)) if problem.m
             else np.zeros((0, problem.n)))
    except Exception as error:
        return dict(accepted=False, reason=f"evaluation failed: {type(error).__name__}")
    if not (np.isfinite(f) and np.all(np.isfinite(c)) and np.all(np.isfinite(g))
            and np.all(np.isfinite(J))):
        return dict(accepted=False, reason="nonfinite values")
    value = np.concatenate([c, x])
    lower = np.concatenate([problem.clower, problem.lower])
    upper = np.concatenate([problem.cupper, problem.upper])
    gradients = np.vstack([J, np.eye(problem.n)])
    with np.errstate(invalid="ignore"):
        primal = float(max(0.0, np.max(np.where(np.isfinite(lower), lower - value, 0.0)),
                           np.max(np.where(np.isfinite(upper), value - upper, 0.0))))
    scale = max(1.0, float(np.max(np.abs(g))) if g.size else 1.0)
    limit = DUAL * scale

    def passes(cert):
        return (cert["stationarity"] <= limit and cert["complementarity"] <= limit
                and cert["dual_sign"] <= limit)

    candidates = []
    if constraint_multipliers is not None and variable_multipliers is not None:
        own = np.concatenate([np.asarray(constraint_multipliers, float).reshape(-1),
                              np.asarray(variable_multipliers, float).reshape(-1)])
        if own.size == value.size and np.all(np.isfinite(own)):
            candidates.append(("solver", _certificate(g, gradients, value, lower, upper, own)))
    if not candidates or not passes(candidates[0][1]):
        def near(v, b):
            return math.isfinite(b) and abs(v - b) <= 1e-6 * (1 + abs(b))

        active = [i for i, (v, lo, up) in enumerate(zip(value, lower, upper))
                  if lo == up or near(v, lo) or near(v, up)]
        mu = np.zeros(value.size)
        if active:
            mu[active] = np.linalg.lstsq(gradients[active].T, g, rcond=None)[0]
        candidates.append(("least-squares", _certificate(g, gradients, value, lower, upper, mu)))
    source, cert = next(((s, c) for s, c in candidates if passes(c)), candidates[-1])
    return dict(accepted=bool(primal <= PRIMAL and passes(cert)), objective=float(f),
                primal=primal, **cert, gradient_scale=scale, certificate=source)
