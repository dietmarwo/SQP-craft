"""Restoration through the compiled extension, including exceptional exits."""
import importlib.util
import pathlib

import pytest
from sqp_craft import (Cancellation, EvaluationError, ProblemSpec, SqpOptions,
                       UndefinedPointError, kkt_residuals, solve)


def inconsistent_spec():
    return ProblemSpec(lower=[0., 4.], upper=[3., 4.],
                       constraint_lower=[1., 2.], constraint_upper=[1., 2.],
                       jacobian_indptr=[0, 2, 4], jacobian_indices=[0, 1, 0, 1])


class Inconsistent:
    def __init__(self, undefined=None, fatal=None, cancel=None):
        self.last = None
        self.undefined, self.fatal, self.cancel = undefined, fatal, cancel

    def functions(self, x):
        assert 0 <= x[0] <= 3 and x[1] == 4
        self.last = x
        if x[0] > .75 and self.undefined == 'functions':
            raise UndefinedPointError('trial domain')
        if x[0] > 0 and self.fatal is not None:
            raise self.fatal
        if x[0] > 0 and self.cancel is not None:
            self.cancel.cancel()
        return x[0]**2, [x[0], x[0]]

    def derivatives(self, x):
        assert x == self.last
        if x[0] > .75 and self.undefined == 'derivatives':
            raise UndefinedPointError('trial derivative domain')
        return [2*x[0], 0.], [1., 0., 1., 0.]


def run(model=None, **settings):
    return solve(inconsistent_spec(), model or Inconsistent(), [0., 4.], method='sqp',
                 method_options=SqpOptions(**settings))


def test_hs071_recovery_and_independent_residuals():
    path = pathlib.Path(__file__).parent.parent / 'examples' / 'restore_hs071.py'
    module_spec = importlib.util.spec_from_file_location('restore_example', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = module.specification()
    events = []
    result = solve(spec, module.Hs071(), [1.]*4, method='sqp', observer=lambda p: events.append(p.iteration),
                   method_options=SqpOptions(record_history=True))
    assert result.status == 'converged'
    assert abs(result.evaluation.objective - 17.0140173) < 1e-6
    residual = kkt_residuals(spec, result.candidate, result.evaluation, result.multipliers)
    assert residual.primal <= 1e-7 and residual.stationarity <= 1e-6
    assert residual.complementarity <= 1e-6 and residual.dual_sign <= 1e-8
    assert events == list(range(result.iterations+1))
    assert result.report.restoration_steps > 0
    assert len(result.report.restoration_history) == result.report.restoration_steps
    assert all(h.merit_after < h.merit_before for h in result.report.restoration_history)
    disabled = solve(spec, module.Hs071(), [1.]*4, method='sqp',
                     method_options=SqpOptions(max_restoration_steps=0))
    assert disabled.status == 'linearization_infeasible'


@pytest.mark.parametrize('phase', ['functions', 'derivatives'])
def test_undefined_trials_recover_without_stale_python_exception(phase):
    result = run(Inconsistent(undefined=phase), max_iterations=1, record_history=True)
    assert result.status == 'iteration_limit'
    assert result.report.restoration_steps == 1
    assert result.report.undefined_trials >= 1
    assert 0 < result.candidate[0] <= .75 and result.candidate[1] == 4
    assert result.multipliers is None and result.stationarity is None
    assert result.evaluation.constraints == [result.candidate[0]]*2


def test_fatal_exception_identity_and_cancellation_during_restoration():
    fatal = LookupError('model error during restoration')
    with pytest.raises(LookupError) as caught:
        run(Inconsistent(fatal=fatal))
    assert caught.value is fatal
    class RecoveredThenFatal(Inconsistent):
        def functions(self, x):
            if x[0] > .75:
                raise UndefinedPointError('recover this first')
            return super().functions(x)
    with pytest.raises(LookupError) as caught:
        run(RecoveredThenFatal(fatal=fatal))
    assert caught.value is fatal
    token = Cancellation()
    result = solve(inconsistent_spec(), Inconsistent(cancel=token), [0., 4.], method='sqp', cancellation=token)
    assert result.status == 'cancelled'
    assert result.candidate == [0., 4.]
    assert result.report.restoration_steps == 0


def test_partial_callback_budget_and_restoration_limit():
    result = run(max_callback_calls=3)
    assert result.status == 'evaluation_limit'
    assert result.counts.functions == 2 and result.counts.derivatives == 1
    assert result.candidate == [0., 4.]
    assert result.evaluation.constraints == [0., 0.]
    result = run(max_restoration_steps=1)
    assert result.status == 'restoration_limit'
    assert result.iterations == result.report.restoration_steps == 1
    assert not result.evaluation.feasible_candidate
    assert result.multipliers is None


def test_unacceptable_domain_and_nonfinite_output_do_not_become_feasibility():
    class Undefined(Inconsistent):
        def functions(self, x):
            if x[0] > 0:
                raise UndefinedPointError('no domain here')
            return super().functions(x)
    result = run(Undefined(), max_backtracks=3)
    assert result.status == 'restoration_failed' and result.candidate == [0., 4.]
    assert result.report.undefined_trials == 4
    class Nonfinite(Inconsistent):
        def functions(self, x):
            if x[0] > 0:
                return float('nan'), [x[0]]*2
            return super().functions(x)
    with pytest.raises(EvaluationError):
        run(Nonfinite())


@pytest.mark.parametrize('setting', ['restoration_regularization', 'restoration_step_limit'])
def test_options_reject_invalid_settings_before_model_calls(setting):
    class MustNotRun:
        def functions(self, x): raise AssertionError('callback ran')
        def derivatives(self, x): raise AssertionError('callback ran')
    for invalid in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match=setting):
            run(MustNotRun(), **{setting: invalid})


def test_stationary_point_of_the_violation_ends_restoration_promptly():
    # x = 1 and x = 2 at once: psi = 0.5((x-1)^2 + (x-2)^2) is smallest at x = 1.5.
    # Restoration must stop there as `restoration_failed`, not creep until the
    # budget runs out and report `restoration_limit`.
    for budget in (10, 100):
        result = run(max_restoration_steps=budget, record_history=True)
        assert result.status == 'restoration_failed', result.message
        assert 'stationary point' in result.message
        assert result.report.restoration_steps == 2
        assert result.candidate[0] == pytest.approx(1.5, abs=1e-3)
        assert result.report.restoration_history[-1].merit_after == pytest.approx(0.25, abs=1e-6)
        assert result.multipliers is None and not result.evaluation.feasible_candidate
