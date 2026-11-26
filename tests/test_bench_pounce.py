"""Curvature, status and multiplier checks for the optional comparator."""
from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import pounce_adapter, pounce_fixtures, pounce_report, report, run, s2mpj, solvers, worker
from bench.adapter import S2mpjProblem
from bench.check import check


@pytest.fixture
def wheel():
    try:
        return pounce_adapter.require_wheel()
    except ImportError as error:
        pytest.skip(str(error))


class Quadratic:
    """Model with analytic objective, row derivatives and selectable bounds."""

    name, n = "analytic", 2
    x0 = np.zeros(2)

    def __init__(self, *, target=(0.0, 0.0), weights=(1.0, 3.0),
                 box=([-np.inf, -np.inf], [np.inf, np.inf]), row=None, curved=False,
                 objective=True):
        self.target = np.array(target)
        self.weights = np.array(weights)
        self.xlower, self.xupper = np.array(box)
        self.curved = curved
        self.objgrps = [0] if objective else []
        self.m = int(row is not None)
        if row is not None:
            self.clower, self.cupper = row

    def fx(self, x):
        return self.weights @ (x.ravel() - self.target) ** 2

    def fgx(self, x):
        return self.fx(x), 2.0 * self.weights * (x.ravel() - self.target)

    def fgHx(self, x):
        f, g = self.fgx(x)
        return f, g, np.diag(2.0 * self.weights)

    def cx(self, x):
        u, v = x.ravel()
        return np.array([u * v + v * v if self.curved else u])

    def cJx(self, x):
        u, v = x.ravel()
        return self.cx(x), np.array([[v, u + 2.0 * v] if self.curved else [1.0, 0.0]])

    def cJHx(self, x):
        c, J = self.cJx(x)
        H = np.array([[0.0, 1.0], [1.0, 2.0]]) if self.curved else np.zeros((2, 2))
        return c, J, [H]

    def LgHxy(self, x, y):
        f, g, H = self.fgHx(x)
        c, J, (Hc,) = self.cJHx(x)
        y = y.item()
        return f + y * c.item(), g + y * J[0], H + y * Hc


def test_weighted_hessian_and_split_cache():
    p = S2mpjProblem(Quadratic(row=([-2.0], [2.0]), curved=True))
    model = pounce_adapter.Exact(p)
    x = np.array([0.7, -0.3])
    rows, cols = model.hessianstructure()
    np.testing.assert_array_equal(rows, [0, 1, 1])
    np.testing.assert_array_equal(cols, [0, 0, 1])
    for sigma in (0.0, 0.5, 1.0):
        for y in (-2.0, 0.0, 3.0):
            values = model.hessian(x, np.array([y]), sigma)
            np.testing.assert_allclose(values, [2 * sigma, y, 6 * sigma + 2 * y])
            H = np.array([[values[0], values[1]], [values[1], values[2]]])

            def grad(point):
                J = p.dense_jacobian(p.jacobian_values(point))
                return sigma * p.gradient(point) + y * J[0]

            for v in (np.array([1.0, 2.0]), np.array([-3.0, 0.5])):
                oracle = (grad(x + 1e-6 * v) - grad(x - 1e-6 * v)) / 2e-6
                np.testing.assert_allclose(H @ v, oracle, rtol=1e-6, atol=1e-8)
                if y:
                    wrong = H.copy()
                    wrong[0, 1] = wrong[1, 0] = -y
                    assert not np.allclose(wrong @ v, oracle, rtol=1e-6, atol=1e-8)
    assert model.work.evaluations["hessians"] == 9
    assert model.work.s2mpj_hessian_calls == {"LgHxy": 3, "fgHx": 3, "cJHx": 4}

    model.objective(x)
    model.constraints(x.copy())
    model.gradient(x)
    model.jacobian(x.copy())
    assert model.work.requests == dict(hessian=9, objective=1, constraints=1, gradient=1, jacobian=1)
    assert model.work.evaluations == dict(hessians=9, functions=1, derivatives=1)
    assert p.counts == (1, 1)


def test_hessian_with_zero_objective_and_without_rows():
    no_objective = S2mpjProblem(Quadratic(row=([-2.0], [2.0]), curved=True, objective=False))
    model = pounce_adapter.Exact(no_objective)
    np.testing.assert_allclose(model.hessian(no_objective.x0, np.array([3.0]), 0.5), [0, 3, 6])
    free = S2mpjProblem(Quadratic())
    np.testing.assert_allclose(pounce_adapter.Exact(free).hessian(free.x0, np.zeros(0), 0.5), [1, 0, 3])


@pytest.mark.parametrize("kind", ["upper bound", "lower bound", "upper row", "lower row",
                                  "two-sided row", "equality", "fixed variable", "inactive row"])
def test_multiplier_conversion_and_bounds(wheel, kind):
    target, expected, relevant = 2.0, -1.0, "constraint_multipliers"
    settings = {}
    if "row" in kind or kind == "equality":
        settings["row"] = {
            "upper row": ([-np.inf], [1.0]), "lower row": ([-1.0], [np.inf]),
            "two-sided row": ([-1.0], [1.0]), "equality": ([1.0], [1.0]),
            "inactive row": ([-1.0], [3.0]),
        }[kind]
    else:
        relevant = "variable_multipliers"
        settings["box"] = {
            "upper bound": ([-np.inf, -np.inf], [1.0, np.inf]),
            "lower bound": ([-1.0, -np.inf], [np.inf, np.inf]),
            "fixed variable": ([1.0, -np.inf], [1.0, np.inf]),
        }[kind]
    answer = 1.0
    if kind.startswith("lower"):
        target, expected, answer = -2.0, 1.0, -1.0
    if kind == "inactive row":
        expected, answer = 0.0, 2.0
    p = S2mpjProblem(Quadratic(target=(target, 1.0), weights=(0.5, 1.0), **settings))
    solved = solvers.run("pounce-sqp-exact", p)
    assert solved["claimed"], solved
    assert solved["x"] == pytest.approx([answer, 1.0], abs=1e-7)
    lam, z = solved["constraint_multipliers"], solved["variable_multipliers"]
    independent = check(p, solved["x"], lam, z, include_solver_certificate=True)
    assert independent["accepted"]
    if kind == "fixed variable":
        # In this release fixed variables are eliminated and their duals
        # come back as zero. The checker takes the value stationarity requires
        # and then credits the solver's certificate for the other variables.
        assert independent["certificate"] == "solver"
        assert independent["solver_certificate"]["completed_fixed_multipliers"] == 1
        assert z[0] == 0.0
    else:
        assert solved[relevant][0] == pytest.approx(expected, abs=1e-7)
        assert independent["solver_certificate"]["duals_passed"]
        assert independent["solver_certificate"]["completed_fixed_multipliers"] == 0
    if kind == "upper row":
        reversed_dual = check(p, solved["x"], [-lam[0]], z, include_solver_certificate=True)
        assert reversed_dual["accepted"] and reversed_dual["certificate"] == "least-squares"
        assert not reversed_dual["solver_certificate"]["duals_passed"]


def test_second_coordinate_bound_dual_replay(wheel):
    p = S2mpjProblem(Quadratic(target=(0.0, -2.0), weights=(1.0, 0.5),
                              box=([-np.inf, -1.0], [np.inf, np.inf])))
    for arm in ("pounce-sqp-exact", "pounce-sqp-lbfgs"):
        row = solvers.run(arm, p)
        assert row["claimed"] and row["x"] == pytest.approx([0.0, -1.0])
        assert row["raw_multipliers"]["mult_x_L"] == [0.0, 0.0]
        verified = check(p, row["x"], [], row["variable_multipliers"], include_solver_certificate=True)
        assert verified["accepted"] and verified["certificate"] == "least-squares"
        assert verified["solver_certificate"]["stationarity"] == pytest.approx(1.0)


def test_sqp_dispatch_and_termination(wheel):
    p = S2mpjProblem(Quadratic(target=(2.0, 1.0)))
    for arm in pounce_adapter.MODES:
        row = solvers.run(arm, p)
        assert row["claimed"] and row["report"]["qp_solves"] > 0
        assert row["options"]["sqp_hessian"] == pounce_adapter.MODES[arm]
        assert (row["hessians"] > 0) == (arm == "pounce-sqp-exact")
    limited = pounce_adapter.solve(p, "exact", max_iterations=0)
    assert limited["status_code"] == -1 and not limited["claimed"]
    initial = pounce_adapter.solve(S2mpjProblem(Quadratic()), "exact", max_iterations=5)
    assert initial["claimed"] and initial["iterations"] == 0
    acceptable = pounce_adapter.status_fields(dict(status=1, status_msg="Solved_To_Acceptable_Level"))
    assert acceptable["acceptable"] and not acceptable["claimed"]


def test_missing_comparator_and_downgrade(wheel, monkeypatch):
    with monkeypatch.context() as patch:
        def no_distribution(_):
            raise PackageNotFoundError("pounce-solver")

        patch.setattr(pounce_adapter, "version", no_distribution)
        assert not solvers.available("pounce-sqp-exact")
        assert solvers.available("auto")
    monkeypatch.setattr(pounce_adapter, "Exact", pounce_adapter.FirstDerivatives)
    p = S2mpjProblem(Quadratic(target=(2.0, 1.0)))
    row = solvers.run("pounce-sqp-exact", p)
    assert row["status"] == "error" and row["hessians"] == 0
    assert "exact Hessian was requested" in row["message"]


def test_bad_duals_nonfinite_calls_and_false_success():
    lam, z, raw = pounce_adapter.duals(dict(mult_g=[np.nan], mult_x_L=[0.0], mult_x_U=[0, 0]), 2, 1)
    assert lam is None and z is None
    assert raw["mult_g"] is None and raw["mult_x_L"] is None
    p = S2mpjProblem(Quadratic(box=([-1.0, -1.0], [1.0, 1.0])))
    model = pounce_adapter.Exact(p)
    p.raw.fx = lambda _: np.nan
    with pytest.raises(ValueError, match="nonfinite functions"):
        model.objective(np.array([0.8, 0.7]))
    assert model.work.failures["functions"] == 1
    p.raw.fgHx = lambda _: (0.0, np.zeros(2), np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="nonfinite hessians"):
        model.hessian(p.x0, np.zeros(0), 1.0)
    assert model.work.evaluations["hessians"] == model.work.failures["hessians"] == 1
    p.raw.fx = lambda x: float((x.ravel() ** 2).sum())
    verdict = check(p, [2.0, 0.0])
    rows = report.table([dict(problem="bad", solver="pounce-sqp-exact", claimed=True, check=verdict)])
    assert rows["bad"]["pounce-sqp-exact"]["false_claim"]


def test_release_fixtures(wheel):
    objective = pounce_fixtures.convex_objective(wheel)
    assert objective["passed"] and objective["corrupted_result_rejected"], objective
    sensitivity = pounce_fixtures.curve_fit_influence(wheel)
    assert sensitivity["passed"], sensitivity


def test_selection_and_resume(tmp_path):
    root = ROOT / "docs/benchmarks"
    panel = json.loads((root / "pounce-curvature/selection.json").read_text())
    base = json.loads((root / "selection.json").read_text())
    assert panel["s2mpj_commit"] == base["s2mpj_commit"] == s2mpj.COMMIT
    names = {row["name"] for row in panel["problems"]}
    assert len(names) == len(panel["problems"]) == 20 and set(panel["controls"]) <= names
    by_name = {row["name"]: row for row in base["problems"]}
    assert all(row == by_name[row["name"]] for row in panel["problems"])
    out = tmp_path / "records.jsonl"
    out.write_text('{"event":"environment","configuration_sha256":"A"}\n'
                   '{"event":"result","problem":"HS71","solver":"auto"}\n')
    assert run.done_pairs(out, "A") == {("HS71", "auto")}
    with pytest.raises(ValueError, match="different configuration"):
        run.done_pairs(out, "B")


def test_evidence_worker_solve_deadline(wheel):
    if not (s2mpj.cache_dir() / "S2MPJ/s2mpjlib.py").exists():
        pytest.skip("fetch pinned S2MPJ for worker tests")
    rows = []
    names = ["pounce-sqp-exact", "pounce-sqp-lbfgs"]
    run.solve_problem("HS71", names, 90.0, rows.append, evidence=True, solve_limit=1e-6)
    assert [r["solver"] for r in rows] == names
    assert all(r["status"] == "timeout" and r["message"].startswith("solve exceeded") for r in rows)


def test_a_worker_that_dies_during_a_solve_reports_its_exit_code(monkeypatch):
    class Dead:
        errors = ["Segmentation fault"]

        def __init__(self, name, pending, *, evidence=False):
            self.events = [dict(event="ready", setup_seconds=0.0, dimensions=[2, 0]),
                           dict(event="start", solver=pending[0]), None]
            self.process = self

        def wait(self, timeout=None):
            return -11

        def next(self, timeout):
            return self.events.pop(0)

        def stop(self):
            pass

    monkeypatch.setattr(run, "Worker", Dead)
    rows = []
    run.solve_problem("ANY", ["auto", "clarabel"], 90.0, rows.append)
    assert [(r["solver"], r["status"]) for r in rows] == [("auto", "crash"), ("clarabel", "crash")]
    assert all(r["message"] == "solve: exit -11: Segmentation fault" for r in rows)


def test_zero_objective_factor_preserves_small_constraint_curvature():
    p = S2mpjProblem(Quadratic(weights=(1e20, 1e20), row=([0], [1]), curved=True))
    model = pounce_adapter.Exact(p)
    np.testing.assert_array_equal(model.hessian(p.x0, np.array([1.0]), 0.0), [0, 1, 2])
    assert model.work.s2mpj_hessian_calls == {"cJHx": 1}


def test_original_jacobian_rejects_point_passing_incomplete_solver_pattern():
    class HiddenDerivative(Quadratic):
        # The cubic and its derivative vanish at all three pattern probes.
        def cx(self, x):
            u, v = x.ravel()
            return np.array([u + max(v - 0.5, 0.0) ** 3])

        def cJx(self, x):
            return self.cx(x), np.array([[1.0, 3 * max(x.ravel()[1] - 0.5, 0.0) ** 2]])

    p = S2mpjProblem(HiddenDerivative(target=(-0.5, 1.0), weights=(1.0, 1.0),
                                      row=([0.125], [0.125])))
    x = np.array([0.0, 1.0])
    np.testing.assert_array_equal(p.dense_jacobian(p.jacobian_values(x)), [[1, 0]])
    np.testing.assert_array_equal(p.full_jacobian(x), [[1, 0.75]])
    verdict = check(p, x, [1], [0, 0], include_solver_certificate=True)
    assert verdict["primal"] == 0 and not verdict["accepted"]
    assert verdict["solver_certificate"]["stationarity"] == pytest.approx(0.75)


def test_split_bound_duals_cannot_cancel_a_complementarity_error():
    p = S2mpjProblem(Quadratic(box=([-1, -1], [1, 1])))
    good = check(p, p.x0, [], [0, 0], include_solver_certificate=True)
    assert good["accepted"] and good["certificate"] == "solver"
    row = dict(x=p.x0, constraint_multipliers=[], variable_multipliers=[0, 0],
               raw_multipliers=dict(mult_x_L=[1, 0], mult_x_U=[1, 0]))
    verified = worker.verify(p, row)
    assert verified["accepted"] and verified["certificate"] == "least-squares"
    assert verified["solver_certificate"]["complementarity"] == 1
    assert not verified["solver_certificate"]["duals_passed"]


def fixed_second_variable():
    # Minimum at (1, 0.5) with gradient (0, -9): the fixed variable's multiplier is -9.
    return S2mpjProblem(Quadratic(target=(1.0, 2.0), box=([-np.inf, 0.5], [np.inf, 0.5])))


def test_solver_certificate_completes_missing_multipliers_of_fixed_variables():
    p = fixed_second_variable()
    removed = check(p, [1.0, 0.5], [], [0, 0], include_solver_certificate=True)
    assert removed["accepted"] and removed["certificate"] == "solver"
    assert removed["solver_certificate"]["completed_fixed_multipliers"] == 1
    supplied = check(p, [1.0, 0.5], [], [0, -9], include_solver_certificate=True)
    assert supplied["certificate"] == "solver"
    assert supplied["solver_certificate"]["completed_fixed_multipliers"] == 0
    # A nonzero multiplier is the solver's statement and is checked as supplied.
    wrong = check(p, [1.0, 0.5], [], [0, 1], include_solver_certificate=True)
    assert wrong["accepted"] and wrong["certificate"] == "least-squares"
    assert wrong["solver_certificate"]["stationarity"] == pytest.approx(10)
    assert not wrong["solver_certificate"]["duals_passed"]
    # The completion never covers an error on a free variable.
    off = check(p, [0.9, 0.5], [], [0, 0], include_solver_certificate=True)
    assert not off["accepted"]
    assert off["solver_certificate"]["stationarity"] == pytest.approx(0.2)
    # Both bound arrays of a removed variable are zero, which the side checks accept.
    row = dict(x=[1.0, 0.5], constraint_multipliers=[], variable_multipliers=[0, 0],
               raw_multipliers=dict(mult_x_L=[0, 0], mult_x_U=[0, 0]))
    assert worker.verify(p, row)["certificate"] == "solver"
    # An active bound of a variable that is not fixed gets no such help.
    free = S2mpjProblem(Quadratic(target=(1.0, 2.0), box=([-np.inf, -np.inf], [np.inf, 0.5])))
    missing = check(free, [1.0, 0.5], [], [0, 0], include_solver_certificate=True)
    assert missing["accepted"] and missing["certificate"] == "least-squares"
    assert missing["solver_certificate"]["completed_fixed_multipliers"] == 0


def test_pounce_reports_no_multiplier_for_a_fixed_variable(wheel):
    p = fixed_second_variable()
    for arm in ("pounce-sqp-exact", "pounce-sqp-lbfgs"):
        row = solvers.run(arm, p, evidence=True)
        assert row["claimed"] and row["x"] == pytest.approx([1.0, 0.5])
        assert row["raw_multipliers"]["mult_x_L"][1] == row["raw_multipliers"]["mult_x_U"][1] == 0
        verdict = worker.verify(p, row)
        assert verdict["accepted"] and verdict["certificate"] == "solver"
        assert verdict["solver_certificate"]["completed_fixed_multipliers"] == 1


def test_panel_gate_and_candidate_replay_detect_invalid_evidence(wheel):
    import hashlib

    p = S2mpjProblem(Quadratic(target=(1.0, 0.0)))
    arm = "pounce-sqp-exact"
    row = solvers.run(arm, p, evidence=True)
    row.update(problem=p.name, solver=arm, check=worker.verify(p, row),
               start_sha256=hashlib.sha256(p.x0.astype("<f8").tobytes()).hexdigest())
    row["objective_check"] = worker.objective_check(row["reported_objective"], row["check"]["objective"])
    row = report.table([row])[p.name][arm]
    assert pounce_report.quality_error(row) is None
    pounce_report.recheck_candidate(p, row)
    row["check"]["stationarity"] += 1.0
    with pytest.raises(ValueError, match="stationarity"):
        pounce_report.recheck_candidate(p, row)
    row["raw_multipliers"]["mult_x_L"][0] += 3.0
    with pytest.raises(ValueError, match="dual conversion"):
        pounce_report.recheck_candidate(p, row)
    for objective in (None, float("inf"), float("nan"), 12.0):
        row["objective_check"] = worker.objective_check(objective, 0.0)
        assert "reported objective" in pounce_report.quality_error(row)
    assert "strict success" in pounce_report.quality_error(dict(row, false_claim=True))


def test_configuration_digest_detects_an_edited_resume_header(tmp_path):
    import hashlib

    arm = "pounce-sqp-exact"
    meta = dict(event="environment", cases=["HS71"], solver_order=[arm],
                s2mpj_commit=s2mpj.COMMIT, pounce=pounce_adapter.VERSION,
                pounce_reference_commit=pounce_adapter.COMMIT,
                selection_sha256=hashlib.sha256(pounce_report.SELECTION.read_bytes()).hexdigest())
    meta["configuration_sha256"] = run.configuration_digest(meta)
    result = dict(event="result", problem="HS71", solver=arm, status="timeout")
    path = tmp_path / "panel.jsonl"
    path.write_text("\n".join(map(json.dumps, [meta, result, dict(meta, started="later")])))
    assert pounce_report.read_panel(path, ["HS71"], [arm])[1] == meta
    with path.open("a") as stream:
        stream.write("\n" + json.dumps(dict(meta, pounce="different")))
    with pytest.raises(ValueError, match="digest"):
        pounce_report.read_panel(path, ["HS71"], [arm])
    with pytest.raises(ValueError, match="different configuration"):
        run.done_pairs(path)  # A legacy run cannot append to an evidence file.


def test_native_exact_catenary_recovers_from_flat_first_qp():
    if not (s2mpj.cache_dir() / "S2MPJ/s2mpjlib.py").exists():
        pytest.skip("fetch pinned S2MPJ for the CATENARY regression")
    p = S2mpjProblem(s2mpj.load("CATENARY"))
    row = solvers.run("clarabel+exact", p, evidence=True)
    verdict = worker.verify(p, row)
    assert row["claimed"] and verdict["accepted"], (row["status"], verdict)
    assert verdict["certificate"] == "solver"
    assert verdict["objective"] == pytest.approx(-8184.810369, abs=1e-5)
