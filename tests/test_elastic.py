"""Elastic mode through the compiled Python boundary."""
import importlib.util
import math
import pathlib

import pytest

from sqp_craft import SqpOptions, kkt_residuals, solve
from test_restoration import Inconsistent, inconsistent_spec
from test_sqp import free

ELASTIC = SqpOptions(qp_solver="active-set", elastic_mode=True)


class Square:
    """(x - 2)^2 with x^2 = 1 from x = 0: J = 0, so the linearization 0 = 1 is impossible."""

    def functions(self, x):
        return (x[0] - 2) ** 2, [x[0] ** 2]

    def derivatives(self, x):
        return [2 * (x[0] - 2)], [2 * x[0]]


def test_a_zero_jacobian_start_converges_in_elastic_mode():
    spec = free(1, [1.0], [1.0])
    assert solve(spec, Square(), [0.0], method="sqp").status == "restoration_failed"
    r = solve(spec, Square(), [0.0], method="sqp", method_options=ELASTIC)
    assert r.status == "converged", r.message
    assert r.candidate == pytest.approx([1.0], abs=1e-9)
    assert r.report.elastic_steps == 1 and r.report.restoration_steps == 0
    assert r.report.elastic_weight == pytest.approx(4e4)
    assert r.report.elastic_weight_increases == 0


def test_hs071_needs_no_restoration_in_elastic_mode():
    path = pathlib.Path(__file__).parent.parent / "examples" / "restore_hs071.py"
    module_spec = importlib.util.spec_from_file_location("restore_example", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = module.specification()
    r = solve(spec, module.Hs071(), [1.0] * 4, method="sqp", method_options=ELASTIC)
    assert r.status == "converged", r.message
    assert abs(r.evaluation.objective - 17.0140173) < 1e-6
    assert r.report.restoration_steps == 0 and r.report.elastic_steps >= 1
    k = kkt_residuals(spec, r.candidate, r.evaluation, r.multipliers)
    assert k.primal <= 1e-7 and k.stationarity <= 1e-6
    assert k.complementarity <= 1e-6 and k.dual_sign <= 1e-8


def test_an_inconsistent_problem_is_locally_infeasible_not_converged():
    # x0 = 1 and x0 = 2 cannot both hold.
    r = solve(inconsistent_spec(), Inconsistent(), [0.0, 4.0], method="sqp",
              method_options=ELASTIC)
    assert r.status == "locally_infeasible", r.message
    assert "not established" in r.message
    assert r.report.elastic_weight_increases == 6
    assert r.evaluation.primal_violation >= 0.5 - 1e-9
    assert 1.0 - 1e-9 <= r.candidate[0] <= 2.0 + 1e-9


def test_options_are_checked_before_model_calls():
    class Untouchable:
        def functions(self, x):
            raise AssertionError("model called")

        derivatives = functions

    with pytest.raises(NotImplementedError, match="elastic"):
        solve(free(1, [1.0], [1.0]), Untouchable(), [0.0], method="sqp",
              method_options=SqpOptions(elastic_mode=True))
    for weight in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="elastic_weight"):
            solve(free(1, [1.0], [1.0]), Untouchable(), [0.0], method="sqp",
                  method_options=SqpOptions(qp_solver="active-set", elastic_mode=True,
                                            elastic_weight=weight))


def test_reports_without_elastic_mode_have_no_weight():
    r = solve(free(1, [1.0], [1.0]), Square(), [0.5], method="sqp",
              method_options=SqpOptions(qp_solver="active-set"))
    assert r.report.elastic_weight is None and r.report.elastic_steps == 0
