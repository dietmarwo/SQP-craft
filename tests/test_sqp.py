"""Native-loop acceptance through the compiled Python boundary."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import math
import pytest
from sqp_craft import *


def spec(n=2, lower=None, upper=None, row_lower=(), row_upper=()):
    m = len(row_lower)
    return ProblemSpec(lower=[-math.inf]*n if lower is None else lower, upper=[math.inf]*n if upper is None else upper,
        constraint_lower=row_lower, constraint_upper=row_upper, jacobian_indptr=[i*n for i in range(m+1)], jacobian_indices=list(range(n))*m)


class Quad:
    def __init__(self, row=False): self.row=row; self.last=None
    def functions(self, x):
        self.last=x.copy()
        return (x[0]-1)**2+(x[1]-2)**2, [sum(x)] if self.row else []
    def derivatives(self, x):
        assert x == self.last, "derivatives must follow functions at the same point"
        return [2*(x[0]-1),2*(x[1]-2)], [1.,1.] if self.row else []


@pytest.mark.parametrize("start", [[0.,0.],[-2.,4.],[4.,-3.]])
@pytest.mark.parametrize("equality", [False, True])
def test_analytic_solution_and_independent_residuals(start, equality):
    s = spec(row_lower=[1.] if equality else [], row_upper=[1.] if equality else [])
    r = solve(s, Quad(equality), start, method="sqp", method_options=SqpOptions(record_qp_limit=2))
    assert r.status == "converged", r
    x,y = r.candidate
    assert r.candidate == pytest.approx([0.,1.] if equality else [1.,2.], abs=1e-5)
    assert abs((x-1)**2+(y-2)**2-(2. if equality else 0.)) < 1e-6
    lam = r.multipliers.constraints[0] if equality else 0.
    assert max(abs(2*(x-1)-lam-r.multipliers.variables[0]),abs(2*(y-2)-lam-r.multipliers.variables[1])) < 1e-6
    assert max(vars(kkt_residuals(s,r.candidate,r.evaluation,r.multipliers)).values()) < 1e-6
    assert 0 < len(r.report.qp_trace) <= 2
    assert r.report.qp_backend == "clarabel 0.11.1"


class Circle:
    def functions(self,x): return .5*((x[0]-2)**2+x[1]**2),[x[0]**2+x[1]**2]
    def derivatives(self,x): return [x[0]-2,x[1]],[2*x[0],2*x[1]]


def test_nonlinear_circle_and_derivatives():
    s=spec(row_lower=[-math.inf],row_upper=[1.])
    assert check_derivatives(s,Circle(),[.4,.2]).passed
    r=solve(s,Circle(),[.4,.2],method="sqp")
    assert r.status=="converged",r
    x,y=r.candidate
    assert abs(.5*((x-2)**2+y*y)-.5)<1e-6 and x*x+y*y<=1+1e-7
    assert abs(r.multipliers.constraints[0]+.5)<1e-5
    assert r.iterations>=1


class Domain:
    def __init__(self, fatal=False):self.fatal=fatal;self.last=None
    def functions(self,x):
        self.last=x.copy()
        if x[0]>1.5: raise UndefinedPointError("trial outside propagation domain")
        if self.fatal and x[0]>.1: raise self.fatal
        return (x[0]-1)**2,[]
    def derivatives(self,x):
        assert x==self.last
        return [2*(x[0]-1)],[]


def test_recovered_domain_error_does_not_leak_after_success():
    r=solve(spec(1),Domain(),[0.],method="sqp")
    assert r.status=="converged",r
    assert r.report.undefined_trials>0


def test_later_fatal_error_preserves_identity_after_recovered_error():
    error=RuntimeError("later fatal")
    with pytest.raises(RuntimeError) as caught:solve(spec(1),Domain(error),[0.],method="sqp")
    assert caught.value is error


def test_initial_undefined_keeps_original_exception():
    with pytest.raises(UndefinedPointError,match="propagation"):
        solve(spec(1),Domain(),[2.],method="sqp")


def test_box_domain_and_fixed_coordinate():
    class Box(Quad):
        def functions(self,x):
            assert x[0]==1 and 0<=x[1]<=3
            return super().functions(x)
    r=solve(spec(lower=[1.,0.],upper=[1.,3.]),Box(),[1.,.2],method="sqp")
    assert r.status=="converged",r
    assert r.candidate==pytest.approx([1.,2.],abs=1e-5)
    check=check_derivatives(spec(lower=[1.,0.],upper=[1.,3.]),Box(),[1.,.2])
    assert check.inconclusive and not check.passed and check.skipped==[0]


def test_derivative_checker_detects_sign_and_structural_errors():
    class Wrong(Quad):
        def derivatives(self,x):
            g,j=super().derivatives(x)
            return [-v for v in g],j
    assert not check_derivatives(spec(),Wrong(),[.2,.3]).passed
    class Omission:
        def functions(self,x): return x[0]**2,[x[0]]
        def derivatives(self,x): return [2*x[0]],[]
    s=ProblemSpec(lower=[-2.],upper=[2.],constraint_lower=[0.],constraint_upper=[1.],jacobian_indptr=[0,0],jacobian_indices=[])
    report=check_derivatives(s,Omission(),[.5])
    assert not report.passed and any(i["output"]==1 for i in report.issues)


@pytest.mark.parametrize("budget", [0,1,2,3])
def test_budget_accounting_and_last_complete_point(budget):
    r=solve(spec(),Quad(),[0.,0.],method="sqp",method_options=SqpOptions(max_callback_calls=budget))
    assert r.status=="evaluation_limit"
    assert r.counts.functions+r.counts.derivatives<=budget
    assert r.candidate==(None if budget<2 else [0.,0.])


def test_observer_cancellation_and_no_silent_dispatch():
    r=solve(spec(),Quad(),[0.,0.],method="sqp",observer=lambda p:True)
    assert r.status=="cancelled" and r.candidate==[0.,0.]
    with pytest.raises(NotImplementedError):solve(spec(),Quad(),[0.,0.],method_options=SqpOptions())
    with pytest.raises(ValueError):solve(spec(),Quad(),[0.,0.],method="other")
    r=solve(spec(),Quad(),[0.,0.],method="sqp",method_options=SqpOptions(time_limit_seconds=0))
    assert r.status=="time_limit" and r.counts.functions==0


def worker(start):return solve(spec(),Quad(),start,method="sqp").candidate


def test_process_isolation():
    with ProcessPoolExecutor(2,mp_context=get_context("spawn")) as pool:
        results=list(pool.map(worker,[[0.,0.],[3.,3.]]))
    for x in results:assert x==pytest.approx([1.,2.],abs=1e-5)


class Rosenbrock:
    def functions(self, x): return 100*(x[1]-x[0]*x[0])**2+(1-x[0])**2,[]
    def derivatives(self, x): return [-400*x[0]*(x[1]-x[0]*x[0])+2*(x[0]-1),200*(x[1]-x[0]*x[0])],[]


def test_nonquadratic_requires_multiple_bfgs_steps():
    r=solve(spec(),Rosenbrock(),[-1.2,1.],method="sqp")
    assert r.status=="converged",r
    assert r.iterations>5
    assert r.candidate==pytest.approx([1.,1.],abs=1e-5)
    assert r.evaluation.objective<1e-10
