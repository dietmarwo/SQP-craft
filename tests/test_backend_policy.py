"""Exercise automatic selection through the compiled API, with independent KKT checks."""
import pytest

from sqp_craft import Cancellation, SqpOptions, kkt_residuals, solve
from test_sqp import Quadratic, free, load_example
from test_structured import Chain, chain_spec


def assert_kkt(spec, model, result):
    from dataclasses import replace

    f, c = model.functions(result.candidate)
    g, jac = model.derivatives(result.candidate)
    evaluation = replace(result.evaluation, objective=f, constraints=c,
                         gradient=g, jacobian_values=jac)
    residual = kkt_residuals(spec, result.candidate, evaluation, result.multipliers)
    assert residual.primal <= 1e-7
    assert residual.stationarity <= 1e-6
    assert residual.complementarity <= 1e-6
    assert residual.dual_sign <= 1e-8


@pytest.mark.parametrize("options", [None, SqpOptions(), SqpOptions(hessian_block_size=1)])
def test_omitted_and_dataclass_defaults_choose_dense_elastic(options):
    spec = free()
    r = solve(spec, Quadratic(), [0., 0.], method="sqp", method_options=options)
    assert r.status == "converged"
    assert r.report.qp_solver == "active-set" and r.report.elastic_mode is True
    assert r.report.qp_backend.startswith("goldfarb-idnani")
    assert_kkt(spec, Quadratic(), r)


@pytest.mark.parametrize("backend,elastic", [("auto", False), ("active-set", None),
                                            ("clarabel", None), ("active-set", True)])
def test_explicit_choices_and_elastic_overrides(backend, elastic):
    hs = load_example("restore_hs071")
    spec = hs.specification()
    r = solve(spec, hs.Hs071(), [1.] * 4, method="sqp",
              method_options=SqpOptions(qp_solver=backend, elastic_mode=elastic))
    assert r.status == "converged"
    assert r.report.qp_solver == ("active-set" if backend == "auto" else backend)
    assert r.report.elastic_mode is bool(elastic)
    assert (r.report.restoration_steps == 0) == bool(elastic)
    assert_kkt(spec, hs.Hs071(), r)


def test_default_elastic_replaces_restoration_and_matches_explicit_configuration():
    hs = load_example("restore_hs071")
    spec = hs.specification()
    default = solve(spec, hs.Hs071(), [1.] * 4, method="sqp")
    explicit = solve(spec, hs.Hs071(), [1.] * 4, method="sqp",
                     method_options=SqpOptions(qp_solver="active-set", elastic_mode=True))
    assert default.status == explicit.status == "converged"
    assert default.candidate == explicit.candidate
    assert default.counts == explicit.counts
    assert default.iterations == explicit.iterations
    assert default.report.restoration_steps == 0
    assert default.report.elastic_steps > 0
    assert_kkt(spec, hs.Hs071(), default)


@pytest.mark.parametrize("n", [8, 256, 257, 300])
def test_partitioned_auto_selects_clarabel_on_both_sides_of_dense_limit(n):
    spec = chain_spec(n, objective_pattern=[])
    r = solve(spec, Chain(), [0.05] * n, method="sqp",
              method_options=SqpOptions(hessian="partitioned", qp_reuse_workspace=True))
    assert r.status == "converged", r.message
    assert r.report.qp_solver == "clarabel" and not r.report.elastic_mode
    assert r.report.qp_backend == "clarabel 0.11.1"
    assert r.report.qp_updates > 0
    assert_kkt(spec, Chain(), r)


def test_resolved_choices_survive_cancellation_before_first_qp():
    token = Cancellation()
    token.cancel()
    r = solve(free(), Quadratic(), [0., 0.], method="sqp", cancellation=token)
    assert r.status == "cancelled" and r.report.qp_solves == 0
    assert r.report.qp_solver == "active-set" and r.report.elastic_mode
    dummy = solve(free(), Quadratic(), [0., 0.])
    assert dummy.report.qp_solver is None and not dummy.report.elastic_mode


@pytest.mark.parametrize("options", [SqpOptions(qp_solver="clarabel", elastic_mode=True),
    SqpOptions(hessian="partitioned", elastic_mode=True),
    SqpOptions(hessian="exact", elastic_mode=True), SqpOptions(qp_solver="unknown")])
def test_invalid_combinations_fail_before_callbacks(options):
    class Untouchable:
        def functions(self, x):
            raise AssertionError("model called")
        derivatives = functions

    with pytest.raises((ValueError, NotImplementedError), match="elastic|qp_solver"):
        solve(free(), Untouchable(), [0., 0.], method="sqp", method_options=options)


def test_auto_does_not_change_the_dense_size_limit_or_explicit_large_backend():
    class Untouchable:
        def functions(self, x):
            raise AssertionError("model called")
        derivatives = functions

    spec = chain_spec(257, objective_pattern=[])
    for opts in (SqpOptions(), SqpOptions(hessian="partitioned", qp_solver="active-set")):
        with pytest.raises(NotImplementedError):
            solve(spec, Untouchable(), [0.05] * 257, method="sqp", method_options=opts)


def test_named_benchmarks_pin_backends_and_the_public_default_is_separate():
    from bench.solvers import EXPERIMENTS, SQP_VARIANTS

    assert SQP_VARIANTS["auto"][0] == {}
    for name, (options, _) in (SQP_VARIANTS | EXPERIMENTS).items():
        if name.startswith("clarabel"):
            assert options["qp_solver"] == "clarabel" and options["elastic_mode"] is False
        elif name.startswith("active-set"):
            assert options["qp_solver"] == "active-set"
            assert options["elastic_mode"] == (name in ("active-set+elastic", "active-set+snopt"))
