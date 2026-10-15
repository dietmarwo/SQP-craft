"""Contract tests against the compiled extension (no Python-side solver exists)."""

import math
import pickle

import pytest

from sqp_craft import (
    Cancellation,
    EvaluationError,
    Options,
    ProblemSpec,
    UndefinedPointError,
    solve,
)


class Toy:
    """minimize (x0 - 1)^2 + (x1 - 2)^2 subject to x0 + x1 = 3, 0 <= x <= 4."""

    def __init__(self):
        self.calls = []

    def functions(self, x):
        self.calls.append(("functions", list(x)))
        return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, [x[0] + x[1]]

    def derivatives(self, x):
        self.calls.append(("derivatives", list(x)))
        return [2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)], [1.0, 1.0]


def make_spec(**overrides):
    fields = dict(
        lower=[0.0, 0.0],
        upper=[4.0, 4.0],
        constraint_lower=[3.0],
        constraint_upper=[3.0],
        jacobian_indptr=[0, 2],
        jacobian_indices=[0, 1],
    )
    fields.update(overrides)
    return ProblemSpec(**fields)


@pytest.fixture
def spec():
    return make_spec()


def patched(**methods):
    model = Toy()
    for name, fn in methods.items():
        setattr(model, name, fn)
    return model


# --- dummy behavior -----------------------------------------------------------


def test_evaluates_start_once_and_claims_nothing(spec):
    model = Toy()
    r = solve(spec, model, [1.0, 2.0])
    assert r.status == "stub" and r.iterations == 0
    assert r.candidate == [1.0, 2.0]
    assert r.evaluation.objective == 0.0 and r.evaluation.constraints == [3.0]
    assert r.evaluation.gradient == [0.0, 0.0] and r.evaluation.jacobian_values == [1.0, 1.0]
    assert r.evaluation.feasible_candidate
    assert r.multipliers is None and r.stationarity is None and r.complementarity is None
    assert (r.counts.functions, r.counts.derivatives) == (1, 1)
    assert model.calls == [("functions", [1.0, 2.0]), ("derivatives", [1.0, 2.0])]


def test_constraint_violation_is_reported_not_repaired(spec):
    r = solve(spec, Toy(), [0.0, 0.0])
    assert r.status == "stub" and r.candidate == [0.0, 0.0]
    assert r.evaluation.primal_violation == 3.0 and not r.evaluation.feasible_candidate


def test_start_outside_bounds_is_rejected_before_any_call(spec):
    model = Toy()
    with pytest.raises(ValueError, match=r"start point: entry 0 = -1 lies outside"):
        solve(spec, model, [-1.0, 0.0])
    assert model.calls == []


@pytest.mark.parametrize("x0", [[1.0], [math.inf, 2.0], [math.nan, 2.0]])
def test_malformed_start(spec, x0):
    with pytest.raises(ValueError):
        solve(spec, Toy(), x0)


def test_options(spec):
    with pytest.raises(NotImplementedError, match="independent solves"):
        solve(spec, Toy(), [1.0, 2.0], options=Options(threads=16))
    with pytest.raises(ValueError, match="feasibility_tolerance"):
        solve(spec, Toy(), [1.0, 2.0], options=Options(feasibility_tolerance=0.0))
    loose = solve(spec, Toy(), [0.0, 0.0], options=Options(feasibility_tolerance=5.0))
    assert loose.evaluation.feasible_candidate


def test_unconstrained_model():
    spec = ProblemSpec(
        lower=[-math.inf] * 2,
        upper=[math.inf] * 2,
        constraint_lower=[],
        constraint_upper=[],
        jacobian_indptr=[0],
        jacobian_indices=[],
    )
    model = patched(functions=lambda x: (x[0] ** 2, []), derivatives=lambda x: ([2 * x[0], 0.0], []))
    r = solve(spec, model, [1.0, 2.0])
    assert r.evaluation.constraints == [] and r.evaluation.jacobian_values == []


# --- callback data exchange ---------------------------------------------------


def test_accepts_tuples_and_numpy_arrays(spec):
    np = pytest.importorskip("numpy")

    class NumpyToy:
        def functions(self, x):
            x = np.asarray(x)
            return np.float64(((x - [1.0, 2.0]) ** 2).sum()), np.array([x.sum()])

        def derivatives(self, x):
            return 2.0 * (np.asarray(x) - [1.0, 2.0]), np.ones(2)

    r = solve(spec, NumpyToy(), np.array([0.5, 2.5]))
    assert r.evaluation.gradient == [-1.0, 1.0] and r.evaluation.constraints == [3.0]
    assert solve(spec, Toy(), (1.0, 2.0)).candidate == [1.0, 2.0]


def test_callbacks_receive_fresh_lists(spec):
    def mutate(x):
        x[0] = 100.0
        return 0.0, [3.0]

    original = [1.0, 2.0]
    seen = []
    model = patched(functions=mutate, derivatives=lambda x: (seen.append(list(x)) or ([0.0, 0.0], [1.0, 1.0])))
    r = solve(spec, model, original)
    assert original == [1.0, 2.0] and r.candidate == [1.0, 2.0] and seen == [[1.0, 2.0]]


@pytest.mark.parametrize(
    "method, value, match",
    [
        ("functions", (0.0, []), "constraint values: expected 1 values, got 0"),
        ("functions", (0.0, [1.0, 2.0]), "expected 1 values, got 2"),
        ("functions", (math.nan, [3.0]), "objective is NaN"),
        ("functions", (0.0, [math.inf]), "constraint values: entry 0 is inf"),
        ("derivatives", ([0.0], [1.0, 1.0]), "gradient: expected 2 values, got 1"),
        ("derivatives", ([0.0, 0.0], [1.0]), "Jacobian values: expected 2 values, got 1"),
        ("derivatives", ([math.nan, 0.0], [1.0, 1.0]), "gradient: entry 0 is NaN"),
    ],
)
def test_malformed_outputs_raise_evaluation_error(spec, method, value, match):
    with pytest.raises(EvaluationError, match=match):
        solve(spec, patched(**{method: lambda x: value}), [1.0, 2.0])
    assert issubclass(EvaluationError, RuntimeError)


@pytest.mark.parametrize(
    "value",
    [[0.0, [3.0]], 0.0, (0.0,), (0.0, "3"), ("zero", [3.0]), (None, [3.0])],
)
def test_wrongly_typed_outputs_raise_type_error(spec, value):
    with pytest.raises(TypeError):
        solve(spec, patched(functions=lambda x: value), [1.0, 2.0])


def test_model_exceptions_are_reraised_unchanged(spec):
    class DynamicsError(Exception):
        pass

    def fail(x):
        raise DynamicsError("propagation diverged")

    with pytest.raises(DynamicsError, match="propagation diverged"):
        solve(spec, patched(derivatives=fail), [1.0, 2.0])


def test_undefined_point_is_a_distinct_signal(spec):
    def undefined(x):
        raise UndefinedPointError("negative mass")

    with pytest.raises(UndefinedPointError, match="negative mass"):
        solve(spec, patched(functions=undefined), [1.0, 2.0])
    assert issubclass(UndefinedPointError, EvaluationError)


def test_model_methods_must_be_callable(spec):
    with pytest.raises(TypeError, match="model.functions must be callable"):
        solve(spec, patched(functions=None), [1.0, 2.0])


def test_central_differences_agree_with_values_returned_through_rust(spec):
    model = Toy()
    x, direction, h = [0.8, 1.4], [0.3, -0.7], 1e-6
    plus = [a + h * d for a, d in zip(x, direction)]
    minus = [a - h * d for a, d in zip(x, direction)]
    e = solve(spec, model, x).evaluation
    df = (model.functions(plus)[0] - model.functions(minus)[0]) / (2 * h)
    dc = (model.functions(plus)[1][0] - model.functions(minus)[1][0]) / (2 * h)
    assert sum(g * d for g, d in zip(e.gradient, direction)) == pytest.approx(df, abs=1e-8)
    assert sum(j * d for j, d in zip(e.jacobian_values, direction)) == pytest.approx(dc, abs=1e-8)
    # The check is sensitive: a flipped gradient sign would be caught.
    assert sum(-g * d for g, d in zip(e.gradient, direction)) != pytest.approx(df, abs=1e-8)


# --- cancellation and observer --------------------------------------------------


def test_cancellation_before_start_makes_no_call(spec):
    c = Cancellation()
    c.cancel()
    model = Toy()
    r = solve(spec, model, [1.0, 2.0], cancellation=c)
    assert r.status == "cancelled" and r.candidate is None and r.evaluation is None
    assert model.calls == [] and c.cancelled


def test_cancellation_between_calls(spec):
    c = Cancellation()

    def functions(x):
        c.cancel()
        return 0.0, [3.0]

    model = patched(functions=functions, derivatives=lambda x: pytest.fail("checkpoint skipped"))
    r = solve(spec, model, [1.0, 2.0], cancellation=c)
    assert r.status == "cancelled" and r.candidate is None
    assert (r.counts.functions, r.counts.derivatives) == (1, 0)


def test_print_only_observer_continues(spec, capsys):
    r = solve(spec, Toy(), [1.0, 2.0], observer=print)
    assert r.status == "stub"
    assert "Progress(iteration=0, objective=0.0, primal_violation=0.0)" in capsys.readouterr().out


def test_observer_true_stops_and_keeps_candidate(spec):
    events = []
    r = solve(spec, Toy(), [0.0, 0.0], observer=lambda p: events.append(p) or True)
    assert r.status == "cancelled" and r.candidate == [0.0, 0.0] and r.evaluation is not None
    assert [(p.iteration, p.objective, p.primal_violation) for p in events] == [(0, 5.0, 3.0)]
    assert solve(spec, Toy(), [1.0, 2.0], observer=lambda p: False).status == "stub"


def test_observer_errors(spec):
    with pytest.raises(TypeError, match="observer must return None or False"):
        solve(spec, Toy(), [1.0, 2.0], observer=lambda p: "stop")

    def fail(p):
        raise LookupError("observer failed")

    with pytest.raises(LookupError, match="observer failed"):
        solve(spec, Toy(), [1.0, 2.0], observer=fail)


# --- problem specification ----------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(jacobian_indptr=[0, 3]), "Jacobian pattern: indptr must start at 0 and end"),
        (dict(jacobian_indices=[0, 0]), "strictly increasing"),
        (dict(jacobian_indices=[0, 2]), "column index 2"),
        (dict(constraint_upper=[]), "constraint bounds: expected 1 entries"),
        (dict(lower=[math.nan, 0.0]), "variable bounds: entry 0"),
        (dict(lower=[5.0, 0.0]), "variable bounds: entry 0 has lower=5, upper=4"),
        (dict(lower=[], upper=[], jacobian_indices=[]), "at least one variable"),
    ],
)
def test_invalid_spec_is_rejected_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_spec(**overrides)


def test_spec_rejects_wrong_types():
    with pytest.raises(TypeError):
        make_spec(lower="00")
    with pytest.raises((TypeError, OverflowError)):
        make_spec(jacobian_indices=[0, -1])
    with pytest.raises(TypeError):
        ProblemSpec([0.0], [1.0], [], [], [0], [])  # keyword-only


def test_spec_is_immutable_comparable_and_picklable(spec):
    assert (spec.n, spec.m, spec.nnz) == (2, 1, 2)
    assert repr(spec) == "ProblemSpec(n=2, m=1, nnz=2)"
    assert spec.jacobian_indptr == [0, 2] and spec.constraint_upper == [3.0]
    with pytest.raises(AttributeError):
        spec.lower = [1.0, 1.0]
    clone = pickle.loads(pickle.dumps(spec))
    assert clone == spec and clone is not spec
    assert spec != make_spec(constraint_upper=[4.0])


def test_spec_accepts_scipy_style_numpy_index_arrays():
    np = pytest.importorskip("numpy")
    s = make_spec(
        lower=np.zeros(2),
        jacobian_indptr=np.array([0, 2], dtype=np.int32),
        jacobian_indices=np.array([0, 1], dtype=np.int64),
    )
    assert s.jacobian_indices == [0, 1]
