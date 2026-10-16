"""QP reuse through the compiled Python boundary; no timing assertions."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from sqp_craft import Cancellation, SqpOptions, kkt_residuals, solve
from test_sqp import Circle, Domain, Quadratic, Rosenbrock, free, load_example


def checked(spec, model, start, *, reuse=True, **settings):
    r = solve(spec, model, start, method="sqp",
              method_options=SqpOptions(qp_reuse_workspace=reuse, **settings))
    assert r.status == "converged", r.message
    residual = kkt_residuals(spec, r.candidate, r.evaluation, r.multipliers)
    assert residual.primal <= 1e-7
    assert residual.stationarity <= 1e-6
    assert residual.complementarity <= 1e-6
    assert residual.dual_sign <= 1e-8
    report = r.report
    assert report.qp_setups + report.qp_updates == report.qp_solves + report.qp_cold_retries
    phases = [report.qp_conversion_seconds, report.qp_setup_seconds,
              report.qp_update_seconds, report.qp_solve_seconds, report.qp_check_seconds]
    assert all(t >= 0 for t in phases)
    assert sum(phases) <= report.qp_seconds + 1e-6
    return r


@pytest.mark.parametrize("factory,spec,start", [
    (Quadratic, free(), [0., 0.]),
    (Circle, free(row_lower=[-float("inf")], row_upper=[1.]), [.4, .2]),
    (Rosenbrock, free(), [-1.2, 1.]),
])
def test_opt_in_matches_cold_acceptance(factory, spec, start):
    cold = checked(spec, factory(), start, reuse=False)
    updated = checked(spec, factory(), start)
    assert cold.report.qp_updates == 0 and cold.report.qp_setups == cold.report.qp_solves
    assert updated.report.qp_updates > 0 and updated.report.qp_setups == 1
    assert abs(cold.evaluation.objective - updated.evaluation.objective) < 1e-7


def test_orbit_reuses_one_workspace_and_keeps_original_residuals():
    spec, model, start = load_example("orbit_transfer").build(20)
    r = checked(spec, model, start, max_iterations=300)
    assert abs(r.evaluation.objective + 1.52206) < 1e-5
    assert r.report.qp_setups == 1
    assert r.report.qp_updates == r.report.qp_solves - 1
    assert r.report.qp_cold_retries == 0


def test_restoration_rebuilds_workspace_and_preserves_history():
    hs = load_example("restore_hs071")
    r = checked(hs.specification(), hs.Hs071(), [1.] * 4, record_history=True)
    assert abs(r.evaluation.objective - 17.0140173) < 1e-6
    assert r.report.restoration_steps == 1
    assert r.report.qp_setups >= 3  # Infeasible ordinary QP, restoration, ordinary again.
    assert r.report.qp_updates > 0
    assert len(r.report.history) + len(r.report.restoration_history) == r.iterations + 1


def test_reuse_keeps_undefined_recovery_and_original_fatal_exception():
    r = checked(free(1), Domain(), [0.])
    assert r.report.undefined_trials > 0
    fatal = LookupError("callback identity")
    with pytest.raises(LookupError) as caught:
        checked(free(1), Domain(fatal), [0.])
    assert caught.value is fatal


def test_callback_limit_and_observer_cancel_keep_complete_point():
    limited = solve(free(), Rosenbrock(), [-1.2, 1.], method="sqp",
                    method_options=SqpOptions(qp_reuse_workspace=True, max_callback_calls=15))
    assert limited.status == "evaluation_limit"
    assert limited.counts.functions + limited.counts.derivatives <= 15
    f, c = Rosenbrock().functions(limited.candidate)
    assert limited.evaluation.objective == f and limited.evaluation.constraints == c
    token = Cancellation()
    def observe(p):
        if p.iteration == 2:
            token.cancel()
    stopped = solve(free(), Rosenbrock(), [-1.2, 1.], method="sqp", cancellation=token,
                    observer=observe, method_options=SqpOptions(qp_reuse_workspace=True))
    assert stopped.status == "cancelled" and stopped.iterations == 2
    assert stopped.report.qp_updates > 0


def test_independent_workers_have_independent_workspaces():
    def run(_):
        return checked(free(), Rosenbrock(), [-1.2, 1.])
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(run, range(8)))
    assert all(r.report.qp_setups == 1 for r in results)
    assert all(r.candidate == results[0].candidate for r in results)


def test_default_and_dummy_reports_are_explicit():
    assert SqpOptions().qp_reuse_workspace is False
    r = solve(free(), Quadratic(), [0., 0.])
    assert r.report.qp_setups == r.report.qp_updates == r.report.qp_cold_retries == 0
    assert r.report.qp_retry_reason is None
    assert r.report.qp_setup_seconds == r.report.qp_update_seconds == 0.0
