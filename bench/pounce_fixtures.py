"""Analytic reproductions of the 0.12.0 objective and sensitivity fixes."""
from __future__ import annotations

import json

import numpy as np

from .pounce_adapter import require_wheel
from .worker import objective_check


def convex_objective(pounce):
    def objective(x):
        u, v = x - np.array([2.0, -1.0])
        return 17.0 + u * u + u * v + 2.0 * v * v

    def gradient(x):
        u, v = x - np.array([2.0, -1.0])
        return np.array([2.0 * u + v, u + 4.0 * v])

    result = pounce.minimize(objective, [-10.0, 2.0], jac=gradient,
                             bounds=[(None, 1.0), (None, None)], solver_selection="qp-ipm")
    evaluated = float(objective(result.x))
    reported = float(result.fun)
    # In this fixture f(x) is deterministic. The looser benchmark-wide
    # threshold would miss the particular FD-model drift this test exercises.
    passed = (result.success and result.info["solver"] == "qp-ipm"
              and np.allclose(result.x, [1.0, -0.75], atol=1e-7, rtol=0)
              and abs(evaluated - 17.875) <= 1e-7 and abs(reported - evaluated) <= 1e-12
              and abs(float(result.info["obj_val"]) - evaluated) <= 1e-12)
    return dict(id="939", passed=bool(passed), route=result.info["solver"], x=result.x.tolist(),
                expected_x=[1.0, -0.75], expected_objective=17.875,
                reported=reported, evaluated=evaluated, model=result.info.get("obj_model"),
                corrupted_result_rejected=objective_check(reported + 1.0, evaluated)["mismatch"])


def curve_fit_influence(pounce):
    groups = np.array([0, 0, 1, 1])
    observations = np.array([2.0, 2.0, 3.0, 3.0])
    design = np.column_stack((groups == 0, groups == 1)).astype(float)

    def means(group, a, b):
        return np.where(group == 0, a, b)

    def jacobian(group, a, b):
        return design

    def fit(data, settings, sensitivity):
        return pounce.curve_fit(means, groups, data, p0=[0.7, 1.6], jac=jacobian,
                                options=dict(tol=1e-12, bound_relax_factor=0.0),
                                sensitivity=sensitivity, **settings)

    cases = []
    for label, expected_x, expected, normal, cap in (
        ("active row", [1.0, 2.0], [0.25, -0.25], np.ones(2), 3.0),
        ("active bound", [1.0, 3.0], [0.0, 0.0], np.array([1.0, 0.0]), 1.0),
        ("slack row", [2.0, 3.0], [0.5, 0.0], np.ones(2), 6.0),
    ):
        if label == "active bound":
            settings = dict(bounds=[(None, cap), (None, None)])
        else:
            settings = dict(constraints=[dict(type="ineq", fun=lambda x: cap - normal @ x,
                                              jac=lambda x: -normal)])
        result = fit(observations, settings, True)
        perturbation = np.array([1e-4, 0.0, 0.0, 0.0])
        plus = fit(observations + perturbation, settings, False)
        minus = fit(observations - perturbation, settings, False)
        finite_difference = (plus.popt - minus.popt) / (2e-4)
        slacks = [float(cap - normal @ run.popt) for run in (result, plus, minus)]
        same_activity = (min(slacks) > 0.5 if label == "slack row"
                         else max(abs(slack) for slack in slacks) <= 1e-7)
        influence = result.dpopt_ddata[:, 0]
        passed = (
            result.success and plus.success and minus.success and same_activity
            and np.allclose(result.popt, expected_x, atol=1e-7, rtol=0)
            and np.allclose(influence, expected, atol=1e-4, rtol=0)
            and np.allclose(finite_difference, expected, atol=1e-4, rtol=0)
            and np.allclose(influence, finite_difference, atol=1e-4, rtol=0)
        )
        cases.append(dict(case=label, passed=bool(passed), x=result.popt.tolist(),
                          expected=expected, reported=influence.tolist(),
                          finite_difference=finite_difference.tolist(), slacks=slacks))
    return dict(id="922", passed=all(case["passed"] for case in cases), cases=cases)


def main():
    pounce = require_wheel()
    fixtures = [convex_objective(pounce), curve_fit_influence(pounce)]
    print(json.dumps(fixtures, indent=2))
    if not all(fixture["passed"] for fixture in fixtures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
