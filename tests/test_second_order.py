"""The second-order correction through the compiled Python boundary."""
import math

import pytest

from sqp_craft import SqpOptions, kkt_residuals, solve
from test_sqp import Quadratic, free


class Maratos:
    """Nocedal & Wright, Example 15.4: min 2(x1² + x2² − 1) − x1 s.t. x1² + x2² = 1."""

    def functions(self, x):
        return 2 * (x[0] ** 2 + x[1] ** 2 - 1) - x[0], [x[0] ** 2 + x[1] ** 2]

    def derivatives(self, x):
        return [4 * x[0] - 1, 4 * x[1]], [2 * x[0], 2 * x[1]]


@pytest.mark.parametrize("qp_solver", ["clarabel", "active-set"])
def test_corrections_save_backtracking_on_the_maratos_example(qp_solver):
    spec = free(2, [1.0], [1.0])
    start = [math.cos(0.3), math.sin(0.3)]
    runs = [solve(spec, Maratos(), start, method="sqp",
                  method_options=SqpOptions(qp_solver=qp_solver, record_history=True,
                                            second_order_correction=on))
            for on in (False, True)]
    plain, corrected = runs
    for r in runs:
        assert r.status == "converged", r.message
        assert r.candidate == pytest.approx([1.0, 0.0], abs=1e-6)
        k = kkt_residuals(spec, r.candidate, r.evaluation, r.multipliers)
        assert k.primal <= 1e-7 and k.stationarity <= 1e-6
    assert plain.report.second_order_corrections == 0
    assert corrected.report.second_order_steps >= 1
    assert corrected.counts.functions < plain.counts.functions
    assert any(h.second_order_correction and h.step_length == 1.0
               for h in corrected.report.history)


def test_linear_rows_are_never_corrected():
    r = solve(free(row_lower=[1.0], row_upper=[1.0]), Quadratic(True), [4.0, -3.0],
              method="sqp", method_options=SqpOptions(second_order_correction=True))
    assert r.status == "converged" and r.report.second_order_corrections == 0
