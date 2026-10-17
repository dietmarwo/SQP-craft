"""The four options taken from SNOPT through the compiled Python boundary."""
import math
import pickle

import pytest

from sqp_craft import ProblemSpec, SqpOptions, UndefinedPointError, kkt_residuals, solve
from test_second_order import Maratos
from test_sqp import Quadratic, free

BACKENDS = ["clarabel", "active-set"]


def check(spec, r):
    assert r.status == "converged", r.message
    k = kkt_residuals(spec, r.candidate, r.evaluation, r.multipliers)
    assert k.primal <= 1e-7 and k.stationarity <= 1e-6 and k.complementarity <= 1e-6


@pytest.mark.parametrize("qp_solver", BACKENDS)
def test_the_curvature_repair_replaces_damping(qp_solver):
    spec = free(2, [1.0], [1.0])
    start = [1.5 * math.cos(2.0), 1.5 * math.sin(2.0)]
    runs = {update: solve(spec, Maratos(), start, method="sqp",
                          method_options=SqpOptions(qp_solver=qp_solver, hessian_update=update,
                                                    record_history=True))
            for update in ("damped", "augmented")}
    for r in runs.values():
        check(spec, r)
    damped, repaired = runs["damped"], runs["augmented"]
    assert damped.report.bfgs_damped_updates >= 1 and damped.report.bfgs_modified_updates == 0
    assert repaired.report.bfgs_modified_updates >= 1 and repaired.report.bfgs_damped_updates == 0
    kinds = {h.curvature for h in repaired.report.history if h.step_length is not None}
    assert "modified" in kinds and kinds <= {"exact", "modified", "negligible"}
    assert repaired.report.history[-1].curvature is None


@pytest.mark.parametrize("qp_solver", BACKENDS)
def test_the_augmented_lagrangian_merit_takes_unit_steps(qp_solver):
    spec = free(2, [1.0], [1.0])
    start = [math.cos(0.3), math.sin(0.3)]
    l1, smooth = (solve(spec, Maratos(), start, method="sqp",
                        method_options=SqpOptions(qp_solver=qp_solver, merit=merit,
                                                  record_history=True))
                  for merit in ("l1", "augmented-lagrangian"))
    check(spec, l1)
    check(spec, smooth)
    assert smooth.candidate == pytest.approx([1.0, 0.0], abs=1e-6)
    assert all(h.step_length == 1.0 for h in smooth.report.history if h.step_length is not None)
    assert smooth.counts.functions < l1.counts.functions


class Unbounded:
    """min −10x0 subject to x0² + x1² ≤ 1; from the origin J = 0."""

    def functions(self, x):
        return -10 * x[0], [x[0] ** 2 + x[1] ** 2]

    def derivatives(self, x):
        return [-10.0, 0.0], [2 * x[0], 2 * x[1]]


def test_the_violation_limit_bounds_accepted_points():
    spec = free(2, [-math.inf], [1.0])
    runs = [solve(spec, Unbounded(), [0.0, 0.0], method="sqp",
                  method_options=SqpOptions(violation_limit=limit, record_history=True))
            for limit in (None, 10.0)]
    for r in runs:
        check(spec, r)
    free_run, limited = runs
    assert max(h.primal_violation for h in free_run.report.history) == pytest.approx(99.0)
    assert max(h.primal_violation for h in limited.report.history) <= 10.0
    assert limited.report.violation_limit_rejections >= 1


def test_invalid_merit_settings_raise_before_any_model_call():
    class Untouchable:
        def functions(self, x):
            raise AssertionError("model called")

        derivatives = functions

    spec = free(2, [1.0], [1.0])
    for options, match in [
        (SqpOptions(merit="augmented-lagrangian", second_order_correction=True), "second_order"),
        (SqpOptions(violation_limit=0.5), "violation_limit"),
        (SqpOptions(merit="smooth"), "merit"),
        (SqpOptions(hessian_update="bfgs"), "hessian_update"),
        (SqpOptions(scaling="auto"), "scaling"),
    ]:
        with pytest.raises(ValueError, match=match):
            solve(spec, Untouchable(), [1.0, 0.0], method="sqp", method_options=options)


def line(linear_rows=None, lower=-20.0, upper=20.0, row=1.0):
    return ProblemSpec(lower=[lower] * 2, upper=[upper] * 2, constraint_lower=[row],
                       constraint_upper=[row], jacobian_indptr=[0, 2], jacobian_indices=[0, 1],
                       linear_rows=linear_rows)


def test_linear_rows_are_part_of_the_spec():
    spec = line([0])
    assert spec.linear_rows == [0] and line().linear_rows == []
    assert pickle.loads(pickle.dumps(spec)) == spec and spec != line()
    for rows in ([1], [0, 0]):
        with pytest.raises(ValueError, match="linear rows"):
            line(rows)


@pytest.mark.parametrize("qp_solver", BACKENDS)
def test_a_start_off_the_linear_rows_is_projected(qp_solver):
    # Quadratic(True) has the linear row x0 + x1, here fixed at 1.
    r = solve(line([0]), Quadratic(True), [3.0, -1.0], method="sqp",
              method_options=SqpOptions(qp_solver=qp_solver, record_history=True))
    check(line([0]), r)
    assert r.report.linear_projection
    assert all(h.primal_violation <= 1e-9 for h in r.report.history)
    # Inconsistent linear rows and bounds: x0 + x1 = 3 on the unit box.
    r = solve(line([0], 0.0, 1.0, 3.0), Quadratic(True), [0.5, 0.5], method="sqp",
              method_options=SqpOptions(qp_solver=qp_solver))
    assert r.status == "linear_infeasible", r.message
    assert r.candidate == [0.5, 0.5]


class Parabola:
    """min (x − pull)² + (y − ½)² s.t. x ≥ 1 (linear) and y² = x; undefined for x < ½."""

    def __init__(self, pull=0.0):
        self.pull = pull

    def functions(self, x):
        if x[0] < 0.5:
            raise UndefinedPointError("x < 1/2")
        return (x[0] - self.pull) ** 2 + (x[1] - 0.5) ** 2, [x[0], x[1] ** 2 - x[0]]

    def derivatives(self, x):
        return [2 * (x[0] - self.pull), 2 * (x[1] - 0.5)], [1.0, -1.0, 2 * x[1]]


def parabola(linear_rows):
    return ProblemSpec(lower=[-5.0] * 2, upper=[5.0] * 2, constraint_lower=[1.0, 0.0],
                       constraint_upper=[math.inf, 0.0], jacobian_indptr=[0, 1, 3],
                       jacobian_indices=[0, 0, 1], linear_rows=linear_rows)


def test_declared_rows_stay_hard_in_elastic_mode():
    options = SqpOptions(qp_solver="active-set", elastic_mode=True)
    soft = solve(parabola(None), Parabola(), [1.0, 0.0], method="sqp", method_options=options)
    hard = solve(parabola([0]), Parabola(), [1.0, 0.0], method="sqp", method_options=options)
    check(parabola(None), soft)
    check(parabola([0]), hard)
    assert soft.report.undefined_trials >= 1
    assert hard.report.undefined_trials == 0 and hard.iterations == 1
    assert hard.candidate == pytest.approx([1.0, 1.0], abs=1e-9)


def test_a_row_wrongly_declared_linear_raises():
    # The parabola's Jacobian depends on y, which the first step changes.
    with pytest.raises(ValueError, match="row 1 is declared linear"):
        solve(parabola([0, 1]), Parabola(pull=2.0), [1.0, 1.0], method="sqp")


class BadUnits:
    """The Maratos example with x = (10³u, 10⁻³v) and the row times 100."""

    def functions(self, x):
        a, b = 1e3 * x[0], 1e-3 * x[1]
        q = a * a + b * b
        return 2 * (q - 1) - a, [1e2 * q]

    def derivatives(self, x):
        a, b = 1e3 * x[0], 1e-3 * x[1]
        return [1e3 * (4 * a - 1), 1e-3 * 4 * b], [1e2 * 2 * a * 1e3, 1e2 * 2 * b * 1e-3]


@pytest.mark.parametrize("qp_solver", BACKENDS)
def test_jacobian_scaling_undoes_bad_units(qp_solver):
    spec = free(2, [1e2], [1e2])
    start = [1.5e-3 * math.cos(2.0), 1.5e3 * math.sin(2.0)]
    runs = {mode: solve(spec, BadUnits(), start, method="sqp",
                        method_options=SqpOptions(qp_solver=qp_solver, scaling=mode))
            for mode in ("none", "linear", "jacobian")}
    for r in runs.values():
        check(spec, r)
    assert runs["none"].report.variable_scales == []
    # Without declared linear rows "linear" is the identity.
    assert runs["linear"].report.variable_scales == [1.0, 1.0]
    assert runs["linear"].candidate == runs["none"].candidate
    assert runs["jacobian"].report.variable_scales == [2.0 ** -9, 2.0 ** 9]
    assert 2 * runs["jacobian"].iterations < runs["none"].iterations
