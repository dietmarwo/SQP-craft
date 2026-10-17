"""Complementarity problems (examples/stress_mpcc.py): degenerate step QPs."""
import math

import numpy as np
import pytest

from sqp_craft import SqpOptions, solve
from test_sqp import load_example

mpcc = load_example("stress_mpcc")
cb = load_example("compare_baseline")


def case(seed, index):
    rng = np.random.default_rng(seed)
    for _ in range(index):
        mpcc.problem(rng)
    return mpcc.problem(rng)


def accepted(spec, model, start, result):
    check = cb.evaluate(cb.Case("mpcc", spec, model, start, 0.0, math.inf),
                        result.candidate, result.multipliers)
    return (check["primal"] <= cb.PRIMAL and check["stationarity"] <= cb.STATIONARITY
            and check["complementarity"] <= cb.COMPLEMENTARITY and check["dual_sign"] <= cb.SIGN)


def test_a_residual_at_its_rounding_level_is_not_a_convergence_claim():
    # Seed 1, case 80: Clarabel's QP multipliers reach 5e9, and the
    # stationarity residual (9.5e-7 in Rust, 1.2e-6 in NumPy) is rounding.
    # That was once reported as converged; the solve now continues to a
    # point the independent evaluator accepts.
    spec, model, start = case(1, 80)
    r = solve(spec, model(), start, method="sqp", method_options=SqpOptions(max_iterations=300))
    assert r.status != "converged" or accepted(spec, model, start, r), r.message


@pytest.mark.parametrize("elastic", [False, True])
def test_the_active_set_backend_solves_complementarity_problems(elastic):
    rng = np.random.default_rng(7)
    options = SqpOptions(max_iterations=300, qp_solver="active-set", elastic_mode=elastic)
    for k in range(40):
        spec, model, start = mpcc.problem(rng)
        r = solve(spec, model(), start, method="sqp", method_options=options)
        assert r.status == "converged", (k, r.message)
        assert accepted(spec, model, start, r), k
