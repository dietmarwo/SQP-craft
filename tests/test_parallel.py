"""Explicit native worker budgets through the compiled Python boundary."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from sqp_craft import Cancellation, Options, SqpOptions, kkt_residuals, solve
from test_contract import Toy, make_spec


class OwnedToy(Toy):
    def __init__(self):
        super().__init__()
        self.owner = threading.get_ident()

    def functions(self, x):
        assert threading.get_ident() == self.owner
        return super().functions(x)

    def derivatives(self, x):
        assert threading.get_ident() == self.owner
        return super().derivatives(x)


@pytest.mark.parametrize("backend", ["active-set", "clarabel"])
@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_native_widths_keep_callback_observer_and_kkt_contract(backend, threads):
    spec = make_spec()
    model = OwnedToy()
    events = []

    def observe(progress):
        assert threading.get_ident() == model.owner
        events.append(progress)

    r = solve(spec, model, [0., 0.], method="sqp", options=Options(threads=threads),
              method_options=SqpOptions(qp_solver=backend), observer=observe)
    assert r.status == "converged", r.message
    assert r.candidate == pytest.approx([1., 2.], abs=1e-5)
    k = kkt_residuals(spec, r.candidate, r.evaluation, r.multipliers)
    assert k.primal <= 1e-7 and k.stationarity <= 1e-6
    assert k.complementarity <= 1e-6 and k.dual_sign <= 1e-8
    assert len(events) == r.iterations + 1


@pytest.mark.parametrize("backend", ["active-set", "clarabel"])
@pytest.mark.parametrize("callback", ["functions", "derivatives"])
def test_eight_thread_exception_identity(backend, callback):
    error = LookupError("parallel callback identity")
    model = OwnedToy()

    def fail(_):
        assert threading.get_ident() == model.owner
        raise error

    setattr(model, callback, fail)
    with pytest.raises(LookupError) as caught:
        solve(make_spec(), model, [0., 0.], method="sqp", options=Options(threads=8),
              method_options=SqpOptions(qp_solver=backend))
    assert caught.value is error


@pytest.mark.parametrize("backend", ["active-set", "clarabel"])
def test_eight_thread_precancelled_model_is_not_evaluated(backend):
    token = Cancellation()
    token.cancel()
    model = OwnedToy()
    r = solve(make_spec(), model, [0., 0.], method="sqp", options=Options(threads=8),
              method_options=SqpOptions(qp_solver=backend), cancellation=token)
    assert r.status == "cancelled" and r.candidate is None
    assert model.calls == []


def test_dummy_rejects_native_parallel_request():
    with pytest.raises(NotImplementedError, match="dummy"):
        solve(make_spec(), OwnedToy(), [0., 0.], options=Options(threads=8))


def test_eight_python_callers_keep_their_own_models():
    barrier = threading.Barrier(8)

    def work(i):
        model = OwnedToy()
        barrier.wait(timeout=30)
        result = solve(make_spec(), model, [0., 0.], method="sqp",
                       options=Options(threads=1),
                       method_options=SqpOptions(qp_solver="active-set"))
        return i, result.status, result.candidate

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(work, range(8)))
    for i, (index, status, point) in enumerate(results):
        assert index == i and status == "converged"
        assert point == pytest.approx([1., 2.], abs=1e-5)
