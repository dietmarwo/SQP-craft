"""The benchmark harness (bench/): adapter, independent check and report
arithmetic on hand-built problems, without the network. The last test runs
real S2MPJ problems through the worker and is skipped unless the pinned
checkout is in the cache (`python -m bench.s2mpj fetch`)."""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import report  # noqa: E402
from bench.adapter import S2mpjProblem  # noqa: E402
from bench.check import check  # noqa: E402


class Rows:
    """The CSR slice of S2MPJ's linear part `A` that the adapter reads."""

    def __init__(self, dense):
        dense = np.asarray(dense, dtype=float)
        self.shape = dense.shape
        self.indptr, self.indices = [0], []
        for row in dense:
            self.indices.extend(int(j) for j in np.flatnonzero(row))
            self.indptr.append(len(self.indices))
        self.indptr, self.indices = np.array(self.indptr), np.array(self.indices)

    def tocsr(self):
        return self


class Raw:
    """min (x0 − 1)² + (x1 − 2)² + x2²
    s.t. x0 + x1 ≤ 2 (linear), x0·x2 = 0 (element on x0 and x2),
    −1e20 ≤ x0 ≤ 1e20, 0 ≤ x1 ≤ 3, −1 ≤ x2 ≤ 1, start (5, −1, 0.5)."""

    name = "TOY"
    n, m = 3, 2
    pbclass = "C-CQQR2-AN-3-2"

    def __init__(self, hidden_entry=False):
        self.x0 = np.array([[5.0], [-1.0], [0.5]])
        self.xlower = np.array([[-1e20], [0.0], [-1.0]])
        self.xupper = np.array([[1e20], [3.0], [1.0]])
        self.clower = np.array([[-1e20], [0.0]])
        self.cupper = np.array([[2.0], [0.0]])
        self.objgrps = np.array([0])
        self.congrps = np.array([1, 2])
        self.lincons = np.array([0])
        self.A = Rows([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        self.grelt = [[], [], [0]]
        # With `hidden_entry` the element lists only x0, so the structure
        # misses x2 and the evaluated Jacobian must extend the pattern.
        self.elvar = [[0]] if hidden_entry else [[0, 2]]

    def fx(self, x):
        x = x.reshape(-1)
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + x[2] ** 2

    def fgx(self, x):
        x = x.reshape(-1)
        return self.fx(x), np.array([[2 * (x[0] - 1)], [2 * (x[1] - 2)], [2 * x[2]]])

    def cx(self, x):
        x = x.reshape(-1)
        return np.array([[x[0] + x[1]], [x[0] * x[2]]])

    def cJx(self, x):
        x = x.reshape(-1)
        return self.cx(x), np.array([[1.0, 1.0, 0.0], [x[2], 0.0, x[0]]])


def test_adapter_bounds_start_pattern_and_linear_rows():
    p = S2mpjProblem(Raw())
    assert p.lower[0] == -math.inf and p.upper[0] == math.inf
    assert list(p.x0) == [5.0, 0.0, 0.5] and p.start_clipped
    assert list(p.clower) == [-math.inf, 0.0] and list(p.cupper) == [2.0, 0.0]
    assert p.indptr == [0, 2, 4] and p.indices == [0, 1, 0, 2]
    assert not p.pattern_extended and p.linear_rows == [0]
    J = p.dense_jacobian(p.jacobian_values(p.x0))
    assert np.array_equal(J, [[1.0, 1.0, 0.0], [0.5, 0.0, 5.0]])


def test_adapter_extends_a_pattern_the_structure_misses():
    p = S2mpjProblem(Raw(hidden_entry=True))
    assert p.pattern_extended and p.indices == [0, 1, 0, 2]


def test_adapter_counts_distinct_points():
    p = S2mpjProblem(Raw())
    x, y = p.x0, p.x0 + 0.1
    p.functions(x)
    p.functions(x)
    p.derivatives(x)
    p.functions(y)
    assert p.counts == (2, 1)
    p.reset_counts()
    assert p.counts == (0, 0)


def test_adapter_builds_a_valid_spec():
    pytest.importorskip("sqp_craft")
    p = S2mpjProblem(Raw())
    assert p.spec().m == 2
    assert p.spec(linear=True).linear_rows == [0] and not p.spec().linear_rows


# The toy's minimizer: x0·x2 = 0 with x0 + x1 ≤ 2 active.
# With x2 = 0: min (x0 − 1)² + (x1 − 2)² s.t. x0 + x1 = 2 → x = (0.5, 1.5, 0),
# ∇f = (−1, −1, 0) = λ₀(1, 1, 0) + λ₁(0, 0, 0.5) → λ₀ = −1 (upper side), λ₁ = 0.
SOLUTION = np.array([0.5, 1.5, 0.0])


def test_check_accepts_a_kkt_point_with_the_solvers_multipliers():
    verdict = check(S2mpjProblem(Raw()), SOLUTION, [-1.0, 0.0], [0.0, 0.0, 0.0])
    assert verdict["accepted"] and verdict["certificate"] == "solver"
    assert verdict["objective"] == pytest.approx(0.5)


def test_check_rescues_wrong_sign_multipliers_by_least_squares():
    verdict = check(S2mpjProblem(Raw()), SOLUTION, [1.0, 0.0], [0.0, 0.0, 0.0])
    assert verdict["accepted"] and verdict["certificate"] == "least-squares"
    assert check(S2mpjProblem(Raw()), SOLUTION)["accepted"]


def test_check_rejects_non_kkt_and_infeasible_points():
    p = S2mpjProblem(Raw())
    assert not check(p, [0.4, 1.6 - 0.2, 0.0])["accepted"]  # feasible, not stationary
    assert not check(p, [1.0, 2.0, 0.0])["accepted"]  # stationary for f, violates row 0
    assert not check(p, [math.nan, 0.0, 0.0])["accepted"]
    assert not check(p, [1.0, 2.0])["accepted"]  # wrong size


def test_check_with_an_active_upper_bound():
    # min −x0 with x0 ≤ 1: the bound multiplier is −1 at x0 = 1, and x0 = 0.9
    # is no KKT point with any multipliers.
    raw = Raw()
    raw.m = 0
    raw.lincons = np.array([], dtype=int)
    raw.fx = lambda x: -float(x.reshape(-1)[0])
    raw.fgx = lambda x: (raw.fx(x), np.array([[-1.0], [0.0], [0.0]]))
    raw.xlower = np.array([[-1e20], [0.0], [-1.0]])
    raw.xupper = np.array([[1.0], [3.0], [1.0]])
    p = S2mpjProblem(raw)
    assert check(p, [1.0, 1.0, 0.0], [], [-1.0, 0.0, 0.0])["accepted"]
    assert check(p, [1.0, 1.0, 0.0], [], [1.0, 0.0, 0.0])["certificate"] == "least-squares"
    assert not check(p, [0.9, 1.0, 0.0], [], [-1.0, 0.0, 0.0])["accepted"]


def test_shifted_geometric_mean_and_profile():
    assert report.shifted_geometric_mean([10.0, 10.0], 10.0) == pytest.approx(10.0)
    assert report.shifted_geometric_mean([0.0, 30.0], 10.0) == pytest.approx(10.0)
    prof = report.profile({"a": [1.0, 4.0, math.inf], "b": [2.0, 2.0, 3.0]}, [1, 2, math.inf])
    assert prof["a"] == pytest.approx([1 / 3, 2 / 3, 2 / 3])
    assert prof["b"] == pytest.approx([2 / 3, 1.0, 1.0])


def _record(problem, solver, solved, objective, functions, claimed=True):
    return dict(event="result", problem=problem, solver=solver, claimed=claimed,
                status="converged" if solved else "iteration_limit",
                check=dict(accepted=solved, objective=objective, primal=0.0, stationarity=0.0,
                           complementarity=0.0, dual_sign=0.0, gradient_scale=1.0),
                functions=functions, derivatives=functions, iterations=functions,
                seconds=functions / 100)


def test_pairwise_counts_best_objectives_and_false_claims():
    records = [_record("P1", "clarabel", True, 1.0, 10), _record("P1", "active-set", True, 1.0, 5),
               _record("P2", "clarabel", True, 2.0, 10), _record("P2", "active-set", True, 1.0, 20),
               _record("P3", "clarabel", False, 0.0, 10), _record("P3", "active-set", True, 3.0, 8),
               _record("P4", "clarabel", False, 0.0, 10, claimed=False),
               _record("P4", "active-set", False, 0.0, 10, claimed=False)]
    by_problem = report.table(records)
    assert by_problem["P3"]["clarabel"]["false_claim"]
    assert not by_problem["P2"]["clarabel"]["best"] and by_problem["P2"]["active-set"]["best"]
    p = report.pairwise(by_problem, "clarabel", "active-set")
    assert p["only_a"] == [] and p["only_b"] == ["P3"] and p["both"] == 2
    assert (p["wins"], p["losses"], p["ties"]) == (1, 1, 0)
    assert (p["best_a"], p["best_b"]) == (1, 3)
    selection = dict(problems=[dict(name=f"P{k}", n=3, m=1, subset="nonlinear", control=False)
                               for k in range(1, 5)])
    text = report.render(selection, records, [])
    assert "| clarabel | 4 | 2 | 1 |" in text and "Pairwise comparisons" in text


def _s2mpj_available():
    try:
        from bench import s2mpj

        root = s2mpj.cache_dir() / "S2MPJ"
        return (root / "s2mpjlib.py").exists()
    except Exception:
        return False


@pytest.mark.skipif(not _s2mpj_available(), reason="S2MPJ checkout not in the cache")
def test_worker_runs_real_problems_and_survives_a_timeout():
    pytest.importorskip("scipy")
    from bench import run

    lines = []
    run.solve_problem("HS71", ["clarabel", "active-set", "slsqp"], 90.0, lines.append)
    assert [r["solver"] for r in lines] == ["clarabel", "active-set", "slsqp"]
    for r in lines:
        assert r["check"]["accepted"], r
        assert r["check"]["objective"] == pytest.approx(17.0140173, abs=1e-6)
    lines.clear()
    run.solve_problem("HS71", ["clarabel", "active-set"], 1e-6, lines.append)
    assert [(r["solver"], r["status"]) for r in lines] == [("clarabel", "timeout"),
                                                         ("active-set", "timeout")]
