"""Opt-in exact-Hessian variants for SQP4 and SQP6.

Run `python tutorials/_support/exact_hessian.py` with sqp-craft 0.18.0 or later.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from sqp_craft import ProblemSpec, SqpOptions, check_lagrangian_hessian, solve

from numerics import Case, require
from structured import brach_case


def declare(spec, entries):
    n = spec.n
    pairs = sorted({(i, i) for i in range(n)} | set(entries))
    rows, cols = map(np.asarray, zip(*pairs))
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    new = ProblemSpec(lower=spec.lower, upper=spec.upper,
                      constraint_lower=spec.constraint_lower, constraint_upper=spec.constraint_upper,
                      jacobian_indptr=spec.jacobian_indptr, jacobian_indices=spec.jacobian_indices,
                      hessian_indptr=ptr, hessian_indices=cols)
    return new, pairs


def collocation(n_intervals=20):
    case, _ = brach_case(n_intervals)
    k = n_intervals + 1
    t = case.spec.n - 1
    off_diagonal = [(3*k+i, 2*k+i) for i in range(k)] + [(t, j) for j in range(2*k, 4*k)]
    spec, pairs = declare(case.spec, off_diagonal)

    def hessian(x, lam):
        q = np.asarray(lam).reshape(3, n_intervals)
        w = np.pad(q, ((0, 0), (0, 1))) + np.pad(q, ((0, 0), (1, 0)))
        v, theta = np.asarray(x)[2*k:4*k].reshape(2, k)
        sin, cos = np.sin(theta), np.cos(theta)
        a = w[0]*sin + w[1]*cos
        b = w[0]*cos - w[1]*sin
        d = 0.5 / n_intervals
        values = {}
        for i in range(k):
            vi, ti = 2*k+i, 3*k+i
            values[ti, vi] = d * x[t] * b[i]
            values[ti, ti] = -d * x[t] * (v[i]*a[i] + 9.81*w[2,i]*cos[i])
            values[t, vi] = d * a[i]
            values[t, ti] = d * (v[i]*b[i] - 9.81*w[2,i]*sin[i])
        return [values.get(pair, 0.0) for pair in pairs]

    return case, spec, hessian, np.linspace(0.2, 1.0, spec.m)


def kinetics():
    times = np.linspace(0.0, 8.0, 21)
    obs = 0.5 * (np.exp(-0.5*times) - np.exp(-1.5*times))
    dA = np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                   [[-1., 0., 0.], [1., 0., 0.], [0., 0., 0.]],
                   [[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]])

    @lru_cache(maxsize=2)
    def state_derivatives(p):
        # Advance the linear ODE and both variational equations with the
        # same RK4 stages. These are exact derivatives of the RK4 model.
        A = p[1]*dA[1] + p[2]*dA[2]
        y = np.zeros((3, 13))
        y[0, :2] = p[0], 1.0

        def rhs(state):
            out = A @ state
            out[:, 1:4] += np.einsum("jki,i->kj", dA, state[:, 0])
            cross = np.einsum("jki,il->kjl", dA, state[:, 1:4])
            out[:, 4:] += (cross + cross.transpose(0, 2, 1)).reshape(3, 9)
            return out

        sampled = [y[1].copy()]
        for left, right in zip(times, times[1:]):
            steps = max(1, int(np.ceil((right-left)/0.02)))
            h = (right-left)/steps
            for _ in range(steps):
                a = rhs(y)
                b = rhs(y + h*a/2)
                c = rhs(y + h*b/2)
                d = rhs(y + h*c)
                y += h*(a + 2*b + 2*c + d)/6
            sampled.append(y[1].copy())
        return np.asarray(sampled)

    def function(x):
        error = state_derivatives(tuple(x))[:, 0] - obs
        return 0.5 * error @ error, []

    def derivative(x):
        y = state_derivatives(tuple(x))
        return y[:, 1:4].T @ (y[:, 0] - obs), np.zeros((0, 3))

    case = Case([0.05, 0.1, 0.1], [3, 3, 3], [], [], function, derivative, [0.8, 0.4, 1.3])
    spec, pairs = declare(case.spec, [(1, 0), (2, 0), (2, 1)])

    def hessian(x, _lam):
        y = state_derivatives(tuple(x))
        j = y[:, 1:4]
        matrix = j.T @ j + np.einsum("i,ijk->jk", y[:, 0] - obs, y[:, 4:].reshape(-1, 3, 3))
        return [matrix[i, k] for i, k in pairs]

    return case, spec, hessian, []


def compare(case, spec, hessian, test_multipliers):
    case.lagrangian_hessian = hessian
    audit = check_lagrangian_hessian(spec, case, case.start, test_multipliers)
    require(audit.passed, audit)
    results = {}
    for mode in ("full", "exact"):
        r = solve(spec, case, case.start, method="sqp",
                  method_options=SqpOptions(hessian=mode, qp_solver="active-set", max_iterations=500))
        checked = case.check(r.candidate, r.multipliers)
        if r.status == "converged":
            require(checked["first_order"], checked)
        results[mode] = dict(status=r.status, iterations=r.iterations,
                             objective=r.evaluation.objective,
                             first_order=bool(checked["first_order"]), structure=r.report.structure)
    return results


if __name__ == "__main__":
    print(json.dumps(dict(sqp4=compare(*collocation()), sqp6=compare(*kinetics())), indent=2))
