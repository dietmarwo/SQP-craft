"""The dense active-set QP backend through the compiled Python boundary."""
import importlib.util
import pathlib

import pytest

from sqp_craft import SqpOptions, kkt_residuals, solve
from test_performance_options import padded_regression
from test_sqp import Circle, Quadratic, Rosenbrock, dense_spec, free, load_example

ACTIVE_SET = SqpOptions(qp_solver="active-set")


def assert_kkt(spec, result):
    k = kkt_residuals(spec, result.candidate, result.evaluation, result.multipliers)
    assert k.primal <= 1e-7 and k.stationarity <= 1e-6
    assert k.complementarity <= 1e-6 and k.dual_sign <= 1e-8


@pytest.mark.parametrize("start", [[0.0, 0.0], [-2.0, 4.0], [4.0, -3.0]])
@pytest.mark.parametrize("equality", [False, True])
def test_analytic_solutions(start, equality):
    spec = free(row_lower=[1.0], row_upper=[1.0]) if equality else free()
    r = solve(spec, Quadratic(equality), start, method="sqp", method_options=ACTIVE_SET)
    assert r.status == "converged", r.message
    assert r.candidate == pytest.approx([0.0, 1.0] if equality else [1.0, 2.0], abs=1e-5)
    assert r.report.qp_backend.startswith("goldfarb-idnani")
    assert_kkt(spec, r)


def test_signed_multiplier_and_many_bfgs_steps():
    spec = free(row_lower=[-float("inf")], row_upper=[1.0])
    r = solve(spec, Circle(), [0.4, 0.2], method="sqp", method_options=ACTIVE_SET)
    assert r.status == "converged", r.message
    assert r.multipliers.constraints[0] == pytest.approx(-0.5, abs=1e-5)
    r = solve(free(), Rosenbrock(), [-1.2, 1.0], method="sqp", method_options=ACTIVE_SET)
    assert r.status == "converged" and r.candidate == pytest.approx([1.0, 1.0], abs=1e-5)


def test_orbit_transfer_matches_the_interior_point_backend():
    spec, model, x0 = load_example("orbit_transfer").build(20)
    runs = {qp: solve(spec, model, x0, method="sqp",
                      method_options=SqpOptions(max_iterations=200, qp_solver=qp))
            for qp in ("clarabel", "active-set")}
    for r in runs.values():
        assert r.status == "converged", r.message
        assert_kkt(spec, r)
    a, c = runs["active-set"], runs["clarabel"]
    assert a.evaluation.objective == pytest.approx(c.evaluation.objective, abs=1e-8)
    assert a.iterations == pytest.approx(c.iterations, abs=5)


def test_hs071_with_restoration():
    path = pathlib.Path(__file__).parent.parent / "examples" / "restore_hs071.py"
    module_spec = importlib.util.spec_from_file_location("restore_example", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = module.specification()
    r = solve(spec, module.Hs071(), [1.0] * 4, method="sqp", method_options=ACTIVE_SET)
    assert r.status == "converged", r.message
    assert abs(r.evaluation.objective - 17.0140173) < 1e-6
    assert r.report.restoration_steps > 0
    assert_kkt(spec, r)


@pytest.mark.parametrize("extra", [0, 20])
def test_the_padded_regression_converges(extra):
    # With Clarabel this problem rounds B indefinite; exact QP steps take
    # another path, where the backend's Cholesky factor would be the gate.
    spec, model, start = padded_regression(extra)
    r = solve(spec, model, start, method="sqp",
              method_options=SqpOptions(max_iterations=300, qp_solver="active-set"))
    assert r.status == "converged", r.message
    assert r.report.hessian_resets >= r.report.qp_not_positive_definite
    assert_kkt(spec, r)


def test_warm_start_saves_working_set_changes_and_keeps_the_result():
    # Every bound ends up active: the pull toward ±3 is clipped by the box.
    n = 6
    spec = dense_spec([-1.0] * n, [1.0] * n)

    class Pull:
        def functions(self, x):
            return sum((i + 1) * (v - 3 * (-1) ** i) ** 2 + 0.1 * v ** 4
                       for i, v in enumerate(x)), []

        def derivatives(self, x):
            return [2 * (i + 1) * (v - 3 * (-1) ** i) + 0.4 * v ** 3
                    for i, v in enumerate(x)], []

    warm, cold = (solve(spec, Pull(), [0.0] * n, method="sqp",
                        method_options=SqpOptions(qp_solver="active-set", qp_warm_start=w))
                  for w in (True, False))
    for r in (warm, cold):
        assert r.status == "converged", r.message
        assert r.candidate == pytest.approx([1, -1, 1, -1, 1, -1], abs=1e-12)
        assert_kkt(spec, r)
    # A cold QP enters all six bounds; a warm one starts with them.
    assert cold.report.qp_iterations == n * cold.report.qp_solves
    assert warm.report.qp_iterations == n


def test_unknown_qp_solver_is_rejected():
    with pytest.raises(ValueError, match="qp_solver"):
        solve(free(), Quadratic(), [0.0, 0.0], method="sqp",
              method_options=SqpOptions(qp_solver="simplex"))
