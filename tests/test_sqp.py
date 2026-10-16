"""Native SQP through the compiled Python boundary."""

import importlib.util
import math
import pathlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pytest

from sqp_craft import (
    EvaluationError,
    ProblemSpec,
    SqpOptions,
    UndefinedPointError,
    check_derivatives,
    kkt_residuals,
    solve,
)

INF = math.inf


def dense_spec(lower, upper, row_lower=(), row_upper=()):
    n, m = len(lower), len(row_lower)
    return ProblemSpec(
        lower=lower,
        upper=upper,
        constraint_lower=row_lower,
        constraint_upper=row_upper,
        jacobian_indptr=[i * n for i in range(m + 1)],
        jacobian_indices=list(range(n)) * m,
    )


def free(n=2, row_lower=(), row_upper=()):
    return dense_spec([-INF] * n, [INF] * n, row_lower, row_upper)


class Quadratic:
    """(x0 - 1)^2 + (x1 - 2)^2, optionally with the row x0 + x1."""

    def __init__(self, row=False):
        self.row = row
        self.last = None

    def functions(self, x):
        self.last = list(x)
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, [x[0] + x[1]] if self.row else []

    def derivatives(self, x):
        assert x == self.last, "derivatives must follow functions at the same point"
        return [2 * (x[0] - 1), 2 * (x[1] - 2)], [1.0, 1.0] if self.row else []


class Circle:
    """0.5 * ((x0 - 2)^2 + x1^2) with the row x0^2 + x1^2."""

    def functions(self, x):
        return 0.5 * ((x[0] - 2) ** 2 + x[1] ** 2), [x[0] ** 2 + x[1] ** 2]

    def derivatives(self, x):
        return [x[0] - 2, x[1]], [2 * x[0], 2 * x[1]]


class Rosenbrock:
    def functions(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, []

    def derivatives(self, x):
        return [-400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1), 200 * (x[1] - x[0] ** 2)], []


class Domain:
    """(x - 1)^2, undefined for x > 1.5; optionally raises `fatal` once x > 0.1."""

    def __init__(self, fatal=None):
        self.fatal = fatal
        self.last = None

    def functions(self, x):
        self.last = list(x)
        if x[0] > 1.5:
            raise UndefinedPointError("trial outside the propagation domain")
        if self.fatal is not None and x[0] > 0.1:
            raise self.fatal
        return (x[0] - 1) ** 2, []

    def derivatives(self, x):
        assert x == self.last
        return [2 * (x[0] - 1)], []


def load_example(name):
    path = pathlib.Path(__file__).parent.parent / "examples" / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


# --- analytic solutions -------------------------------------------------------


@pytest.mark.parametrize("start", [[0.0, 0.0], [-2.0, 4.0], [4.0, -3.0]])
@pytest.mark.parametrize("equality", [False, True])
def test_analytic_solution_and_independent_residuals(start, equality):
    spec = free(row_lower=[1.0], row_upper=[1.0]) if equality else free()
    r = solve(spec, Quadratic(equality), start, method="sqp",
              method_options=SqpOptions(record_qp_limit=2))
    assert r.status == "converged", r.message
    assert r.candidate == pytest.approx([0.0, 1.0] if equality else [1.0, 2.0], abs=1e-5)
    assert r.evaluation.objective == pytest.approx(2.0 if equality else 0.0, abs=1e-6)
    if equality:
        assert r.multipliers.constraints[0] == pytest.approx(-2.0, abs=1e-5)
    residuals = kkt_residuals(spec, r.candidate, r.evaluation, r.multipliers)
    assert max(vars(residuals).values()) < 1e-6
    assert 0 < len(r.report.qp_trace) <= 2
    assert r.report.qp_backend == "clarabel 0.11.1"


def test_nonlinear_inequality_with_signed_multiplier():
    spec = free(row_lower=[-INF], row_upper=[1.0])
    assert check_derivatives(spec, Circle(), [0.4, 0.2]).passed
    r = solve(spec, Circle(), [0.4, 0.2], method="sqp")
    assert r.status == "converged", r.message
    assert r.evaluation.objective == pytest.approx(0.5, abs=1e-6)
    assert r.evaluation.constraints[0] <= 1 + 1e-7
    assert r.multipliers.constraints[0] == pytest.approx(-0.5, abs=1e-5)


def test_rosenbrock_needs_many_bfgs_steps():
    r = solve(free(), Rosenbrock(), [-1.2, 1.0], method="sqp")
    assert r.status == "converged", r.message
    assert r.iterations > 5
    assert r.candidate == pytest.approx([1.0, 1.0], abs=1e-5)


def test_orbit_transfer_collocation():
    spec, model, x0 = load_example("orbit_transfer").build(20)
    check = check_derivatives(spec, model, x0)
    assert check.passed and check.fixed == [0, 21, 42]
    r = solve(spec, model, x0, method="sqp",
              method_options=SqpOptions(max_iterations=200, record_history=True))
    assert r.status == "converged", r.message
    assert -r.evaluation.objective == pytest.approx(1.52206, abs=5e-5)
    assert r.iterations < 150
    assert len(r.report.history) == r.iterations + 1
    assert r.report.history[-1].step_length is None


def test_inconsistent_linearization_is_not_reported_as_infeasibility():
    class Square:  # x^2 = 1 from x = 0: J = 0, so the linearization 0 = 1 is impossible
        def functions(self, x):
            return (x[0] - 2) ** 2, [x[0] ** 2]

        def derivatives(self, x):
            return [2 * (x[0] - 2)], [2 * x[0]]

    r = solve(free(1, [1.0], [1.0]), Square(), [0.0], method="sqp")
    assert r.status == "linearization_infeasible"
    assert "may still be feasible" in r.message
    assert r.candidate == [0.0]


# --- box contract -------------------------------------------------------------


def test_every_callback_point_stays_in_the_box_and_fixed_variables_stay_fixed():
    class Box(Quadratic):
        def functions(self, x):
            assert x[0] == 1 and 0 <= x[1] <= 3, x
            return super().functions(x)

    spec = dense_spec([1.0, 0.0], [1.0, 3.0])
    r = solve(spec, Box(), [1.0, 0.2], method="sqp")
    assert r.status == "converged", r.message
    assert r.candidate == pytest.approx([1.0, 2.0], abs=1e-5)


# --- callback errors ----------------------------------------------------------


def test_recovered_undefined_trial_does_not_leak_after_success():
    r = solve(free(1), Domain(), [0.0], method="sqp")
    assert r.status == "converged", r.message
    assert r.report.undefined_trials > 0


def test_later_fatal_error_keeps_its_identity_after_a_recovered_one():
    error = RuntimeError("later fatal")
    with pytest.raises(RuntimeError) as caught:
        solve(free(1), Domain(error), [0.0], method="sqp")
    assert caught.value is error


def test_undefined_start_raises_the_original_exception():
    with pytest.raises(UndefinedPointError, match="propagation"):
        solve(free(1), Domain(), [2.0], method="sqp")


def test_nonfinite_output_after_a_recovered_undefined_trial_is_an_evaluation_error():
    class NanAfterDomain(Domain):
        def functions(self, x):
            if 0.1 < x[0] <= 1.5:
                return math.nan, []
            return super().functions(x)

    with pytest.raises(EvaluationError, match="objective") as caught:
        solve(free(1), NanAfterDomain(), [0.0], method="sqp")
    assert not isinstance(caught.value, UndefinedPointError)


# --- derivative checker ---------------------------------------------------------


def test_derivative_checker_finds_sign_and_pattern_errors():
    class WrongSign(Quadratic):
        def derivatives(self, x):
            g, j = super().derivatives(x)
            return [-v for v in g], j

    assert not check_derivatives(free(), WrongSign(), [0.2, 0.3]).passed

    class Omission:  # the row depends on x0, but the pattern declares no entry
        def functions(self, x):
            return x[0] ** 2, [x[0]]

        def derivatives(self, x):
            return [2 * x[0]], []

    spec = ProblemSpec(lower=[-2.0], upper=[2.0], constraint_lower=[0.0], constraint_upper=[1.0],
                       jacobian_indptr=[0, 0], jacobian_indices=[])
    report = check_derivatives(spec, Omission(), [0.5])
    assert not report.passed
    assert [(i.variable, i.output, i.analytic) for i in report.issues] == [(0, 1, 0.0)]


def test_derivative_checker_lists_fixed_variables_without_failing():
    report = check_derivatives(dense_spec([1.0, 0.0], [1.0, 3.0]), Quadratic(), [1.0, 0.2])
    assert report.passed and not report.inconclusive
    assert report.fixed == [0] and report.skipped == [] and report.checked == 1


def test_derivative_checker_skips_undefined_perturbations_without_raising():
    # The perturbation x + h is undefined at x = 1.5. The checker must neither
    # raise the consumed UndefinedPointError nor count the coordinate as checked.
    report = check_derivatives(free(1), Domain(), [1.5])
    assert report.inconclusive and not report.passed
    assert report.skipped == [0] and report.issues == []


# --- budgets, observer, dispatch ------------------------------------------------


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_budget_accounting_keeps_the_last_complete_point(budget):
    r = solve(free(), Quadratic(), [0.0, 0.0], method="sqp",
              method_options=SqpOptions(max_callback_calls=budget))
    assert r.status == "evaluation_limit"
    assert r.counts.functions + r.counts.derivatives <= budget
    assert r.candidate == (None if budget < 2 else [0.0, 0.0])


def test_observer_stop_time_limit_and_explicit_dispatch():
    r = solve(free(), Quadratic(), [0.0, 0.0], method="sqp", observer=lambda p: True)
    assert r.status == "cancelled" and r.candidate == [0.0, 0.0]
    r = solve(free(), Quadratic(), [0.0, 0.0], method="sqp",
              method_options=SqpOptions(time_limit_seconds=0))
    assert r.status == "time_limit" and r.counts.functions == 0
    with pytest.raises(NotImplementedError):
        solve(free(), Quadratic(), [0.0, 0.0], method_options=SqpOptions())
    with pytest.raises(ValueError, match="unknown method"):
        solve(free(), Quadratic(), [0.0, 0.0], method="other")
    with pytest.raises(ValueError, match="record_qp_limit"):
        solve(free(), Quadratic(), [0.0, 0.0], method="sqp",
              method_options=SqpOptions(record_qp_limit=17))


def test_observer_sees_every_accepted_iterate():
    events = []
    r = solve(free(), Rosenbrock(), [-1.2, 1.0], method="sqp", observer=events.append)
    assert r.status == "converged"
    assert [e.iteration for e in events] == list(range(r.iterations + 1))


def _worker(start):
    return solve(free(), Quadratic(), start, method="sqp").candidate


def test_independent_solves_in_worker_processes():
    with ProcessPoolExecutor(2, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(_worker, [[0.0, 0.0], [3.0, 3.0]]))
    for x in results:
        assert x == pytest.approx([1.0, 2.0], abs=1e-5)
