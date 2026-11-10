"""Regression tests for the sparse native and Python entry points."""
import pickle

import numpy as np
import pytest

from sqp_craft import ProblemSpec, SqpOptions, check_lagrangian_hessian, solve

def chain_spec(n, **structure):
    return ProblemSpec(lower=[-1] * n, upper=[1] * n,
                       constraint_lower=[0.2] * (n-1), constraint_upper=[0.2] * (n-1),
                       jacobian_indptr=range(0, 2*n, 2),
                       jacobian_indices=[j for i in range(n-1) for j in (i, i+1)],
                       **structure)


class Chain:
    def functions(self, x):
        x = np.asarray(x)
        return -x[-1], x[:-1] + 0.1*x[1:]**2

    def derivatives(self, x):
        g = np.zeros(len(x))
        g[-1] = -1
        return g, np.column_stack([np.ones(len(x)-1), 0.2*np.asarray(x[1:])]).ravel()


def test_partitioned_connected_chain_uses_the_sparse_path():
    n = 300
    spec = chain_spec(n, objective_pattern=[])
    options = SqpOptions(hessian="partitioned", qp_reuse_workspace=True)
    r = solve(spec, Chain(), [0.05]*n, method="sqp", method_options=options)
    assert r.status == "converged", r.message
    assert r.candidate[-1] == pytest.approx(1.0, abs=1e-7)
    assert max(abs(v-0.2) for v in Chain().functions(r.candidate)[1]) < 1e-7
    info = r.report.structure
    assert info["hessian_nnz"] == info["hessian_factor_nnz"] == 2*n-1
    assert info["qp_factor_nnz"] is not None
    assert not info["fell_back_to_full"]
    assert sum(e["updates"] for e in info["elements"]) > 0
    assert r.report.qp_updates > 0

    for bad in (SqpOptions(hessian="partitioned", qp_solver="active-set"),
                SqpOptions(hessian="partitioned", structured_memory_limit_bytes=1024)):
        with pytest.raises(NotImplementedError):
            solve(spec, Chain(), [0.05]*n, method="sqp", method_options=bad)
    with pytest.raises(NotImplementedError, match="dense pilot limit"):
        solve(spec, Chain(), [0.05]*n, method="sqp")
    with pytest.raises(ValueError, match="explicit objective_pattern"):
        solve(chain_spec(n), Chain(), [0.05]*n, method="sqp", method_options=options)


def test_structure_validates_and_round_trips_through_pickle():
    spec = chain_spec(3, elements=[[1], [0]], objective_pattern=[],
                      hessian_indptr=[0, 1, 2, 3], hessian_indices=[0, 1, 2])
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert spec.elements == [[1], [0]]
    assert spec.objective_pattern == []
    with pytest.raises(ValueError, match="row 0 appears more than once"):
        chain_spec(3, elements=[[0], [0]])
    with pytest.raises(ValueError, match="row 1 is missing"):
        chain_spec(3, elements=[[0]])
    with pytest.raises(ValueError, match="objective_pattern: index 3"):
        chain_spec(3, objective_pattern=[3])
    with pytest.raises(ValueError, match="row 1 must end with its diagonal"):
        chain_spec(3, hessian_indptr=[0, 1, 2, 3], hessian_indices=[0, 0, 2])


def test_exact_hessian_sign_missing_entries_and_convexification():
    class Quadratic:
        def functions(self, x):
            return (x[0] - 1.0)**2, [x[0]*x[1]]

        def derivatives(self, x):
            return [2*(x[0]-1), 0.0], [x[1], x[0]]

        def lagrangian_hessian(self, x, lam):
            return [2.0, -lam[0], 0.0]

    spec = ProblemSpec(lower=[0.2, 0.2], upper=[2., 2.],
                       constraint_lower=[1.0], constraint_upper=[1.0],
                       jacobian_indptr=[0, 2], jacobian_indices=[0, 1],
                       hessian_indptr=[0, 1, 3], hessian_indices=[0, 0, 1])
    model = Quadratic()
    assert check_lagrangian_hessian(spec, model, [0.8, 1.2], [0.6]).passed
    model.lagrangian_hessian = lambda x, lam: [2.0, 0.0, 0.0]
    missing = check_lagrangian_hessian(spec, model, [0.8, 1.2], [0.6])
    assert {(i.variable, i.output) for i in missing.issues} == {(0, 1), (1, 0)}
    model = Quadratic()
    options = SqpOptions(hessian="exact", max_callback_calls=1000)
    r = solve(spec, model, [0.8, 1.2], method="sqp", method_options=options)
    assert r.status == "converged", r.message
    assert np.allclose(r.candidate, [1.0, 1.0], atol=1e-5)
    assert r.report.structure["maximum_diagonal_shift"] > 0
    assert r.report.structure["exact_hessian_calls"] > 0
    limited = solve(spec, model, [0.8, 1.2], method="sqp",
                    method_options=SqpOptions(hessian="exact", max_callback_calls=2))
    assert limited.status == "evaluation_limit"
    assert limited.report.structure["exact_hessian_calls"] == 0


def test_exact_mode_requires_a_callable_hessian():
    class NoHessian:
        def functions(self, x):
            return (x[0] - 1.0)**2, [x[0]*x[1]]

        def derivatives(self, x):
            return [2*(x[0]-1), 0.0], [x[1], x[0]]

    spec = ProblemSpec(lower=[0.2, 0.2], upper=[2., 2.],
                       constraint_lower=[1.0], constraint_upper=[1.0],
                       jacobian_indptr=[0, 2], jacobian_indices=[0, 1],
                       hessian_indptr=[0, 1, 3], hessian_indices=[0, 0, 1])
    options = SqpOptions(hessian="exact")
    with pytest.raises(NotImplementedError, match="lagrangian_hessian"):
        solve(spec, NoHessian(), [0.8, 1.2], method="sqp", method_options=options)
    model = NoHessian()
    model.lagrangian_hessian = [2.0, 0.0, 0.0]
    with pytest.raises(TypeError, match="lagrangian_hessian must be callable"):
        solve(spec, model, [0.8, 1.2], method="sqp", method_options=options)
