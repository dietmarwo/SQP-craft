"""Explicit curvature blocks and the Hessian gate keep the KKT and callback contracts."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sqp_craft import ProblemSpec, SqpOptions, kkt_residuals, solve
from test_sqp import Quadratic, free

REGRESSION = Path(__file__).resolve().parents[1] / "docs/performance/regression-seed4-case86.json"


def assert_kkt(spec, result):
    k = kkt_residuals(spec, result.candidate, result.evaluation, result.multipliers)
    assert k.primal <= 1e-7 and k.stationarity <= 1e-6
    assert k.complementarity <= 1e-6 and k.dual_sign <= 1e-8


@pytest.mark.parametrize("block", [0, 1, 2, 8, 256])
@pytest.mark.parametrize("reuse", [False, True])
def test_explicit_blocks_known_quadratic(block, reuse):
    spec = free()
    options = SqpOptions(hessian_block_size=block, qp_reuse_workspace=reuse)
    r = solve(spec, Quadratic(), [0.0, 0.0], method="sqp", method_options=options)
    assert r.status == "converged" and r.evaluation.objective < 1e-10
    assert_kkt(spec, r)


def test_invalid_block_size_rejected():
    with pytest.raises(ValueError):
        solve(free(), Quadratic(), [0.0, 0.0], method="sqp",
              method_options=SqpOptions(hessian_block_size=257))


def padded_regression(extra: int):
    """Seed 4, case 86 of the stress family, plus `extra` decoupled variables."""
    data = json.loads(REGRESSION.read_text())
    target = np.concatenate([data["target"], np.full(extra, 0.3)])
    rows = [(np.asarray(r["a"]), r["s"], np.asarray(r["y"])) for r in data["rows"]]
    n0, n, m = len(data["target"]), len(data["target"]) + extra, len(rows)

    class Model:
        def functions(self, x):
            x = np.asarray(x)
            c = [float(a @ x[:n0] + s * np.sum((x[:n0] - y) ** 2)) for a, s, y in rows]
            return float(np.sum((x - target) ** 2)), c

        def derivatives(self, x):
            x = np.asarray(x)
            jac = [np.concatenate([a + 2 * s * (x[:n0] - y), np.zeros(extra)]) for a, s, y in rows]
            return list(2 * (x - target)), list(np.concatenate(jac))

    spec = ProblemSpec(
        lower=data["lower"] + [-2.0] * extra, upper=data["upper"] + [2.0] * extra,
        constraint_lower=[-math.inf if v == "-inf" else v for v in data["row_lower"]],
        constraint_upper=data["row_upper"],
        jacobian_indptr=[i * n for i in range(m + 1)], jacobian_indices=list(range(n)) * m)
    return spec, Model(), data["start"] + [1.5] * extra


@pytest.mark.parametrize("extra", [0, 20])
def test_updates_that_round_indefinite_reset_the_hessian(extra):
    # Cancellation in the explicit BFGS terms leaves B indefinite here. With
    # 20 extra variables the delivered rank-updated factor missed it and the
    # solve raised NotImplementedError from the QP adapter's gate.
    spec, model, start = padded_regression(extra)
    r = solve(spec, model, start, method="sqp",
              method_options=SqpOptions(max_iterations=300, qp_reuse_workspace=True))
    assert r.status == "converged", r.message
    assert r.report.hessian_resets >= 1
    assert_kkt(spec, r)
