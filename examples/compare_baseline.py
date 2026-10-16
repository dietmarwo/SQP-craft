"""Compare unchanged analytic models/starts with SciPy SLSQP; no timing parity claim.
Run after `pip install '.[bench]'`: python examples/compare_baseline.py > results.json
"""
import json
import math
from time import perf_counter
import numpy as np
import scipy
from scipy.optimize import minimize
from sqp_craft import ProblemSpec, solve


class Model:
    def __init__(self, kind): self.kind=kind; self.last=None; self.functions_count=0; self.derivatives_count=0
    def functions(self,x):
        self.last=list(x); self.functions_count+=1
        if self.kind=="circle": return .5*((x[0]-2)**2+x[1]**2), [x[0]**2+x[1]**2]
        if self.kind=="bound": return (x[0]+1)**2, []
        if self.kind=="rosenbrock": return 100*(x[1]-x[0]**2)**2+(1-x[0])**2, []
        return (x[0]-1)**2+(x[1]-2)**2, [x[0]+x[1]] if self.kind=="equality" else []
    def derivatives(self,x):
        assert list(x)==self.last
        self.derivatives_count+=1
        if self.kind=="circle": return [x[0]-2,x[1]], [2*x[0],2*x[1]]
        if self.kind=="bound": return [2*(x[0]+1)], []
        if self.kind=="rosenbrock": return [-400*x[0]*(x[1]-x[0]**2)+2*(x[0]-1),200*(x[1]-x[0]**2)], []
        return [2*(x[0]-1),2*(x[1]-2)], [1.,1.] if self.kind=="equality" else []


def independent(kind,x,multipliers=None):
    model=Model(kind); f,c=model.functions(x); g,j=model.derivatives(x)
    n=len(x); columns=[]; sides=[]
    primal=0.
    if kind=="equality":primal=abs(c[0]-1);columns.append(j);sides.append(("equality",c[0]-1))
    if kind=="circle":
        primal=max(0.,c[0]-1)
        if abs(c[0]-1)<1e-5:columns.append(j);sides.append(("upper",1-c[0]))
    if kind=="bound":
        primal=max(0.,-x[0])
        if abs(x[0])<1e-5:columns.append([1.]);sides.append(("lower",x[0]))
    if multipliers is None:
        a=np.array(columns).T if columns else np.empty((n,0))
        mu=np.linalg.lstsq(a,np.array(g),rcond=None)[0] if columns else np.array([])
        stationarity=float(np.max(np.abs(np.array(g)-a@mu)))
        comp=max([abs(m*v) for m,(_,v) in zip(mu,sides)]+[0.])
        sign=max([max(0.,m) if s=="upper" else max(0.,-m) if s=="lower" else 0. for m,(s,_) in zip(mu,sides)]+[0.])
    else:
        lam=multipliers.constraints[0] if c else 0.
        stationarity=max(abs(g[i]-(j[i]*lam if c else 0.)-multipliers.variables[i]) for i in range(n))
        comp=abs(lam*(c[0]-1)) if c else 0.
        sign=max(0.,lam) if kind=="circle" else 0.
        for i,z in enumerate(multipliers.variables):
            if kind=="bound":comp=max(comp,abs(z*x[i]));sign=max(sign,-z)
            else:sign=max(sign,abs(z))
    return dict(objective=float(f),primal=float(primal),stationarity=float(stationarity),complementarity=float(comp),dual_sign=float(sign))


def run():
    records=[]
    cases=[("quadratic",[0.,0.],0.),("quadratic",[-2.,4.],0.),("equality",[0.,0.],2.),("equality",[4.,-3.],2.),("circle",[.4,.2],.5),("circle",[.2,.4],.5),("bound",[1.],1.),("rosenbrock",[-1.2,1.],0.)]
    for kind,start,target in cases:
        n=len(start);rows=1 if kind in ("circle","equality") else 0
        lower=[0.] if kind=="bound" else [-math.inf]*n;upper=[math.inf]*n
        s=ProblemSpec(lower=lower,upper=upper,constraint_lower=[1. if kind=="equality" else -math.inf]*rows,constraint_upper=[1.]*rows,jacobian_indptr=[i*n for i in range(rows+1)],jacobian_indices=list(range(n))*rows)
        model=Model(kind);began=perf_counter();r=solve(s,model,start,method="sqp");seconds=perf_counter()-began
        counts=[model.functions_count,model.derivatives_count]
        residual=independent(kind,r.candidate,r.multipliers) if r.candidate is not None else None
        records.append(dict(case=kind,start=start,solver="sqp-craft",status=r.status,seconds=seconds,functions=counts[0],derivatives=counts[1],iterations=r.iterations,residuals=residual,target=target))
        model=Model(kind)
        def obj(x):return model.functions(x)[0]
        def grad(x):model.functions(x);return np.array(model.derivatives(x)[0])
        constraints=[]
        if rows:
            sign=1 if kind=="equality" else -1
            def fun(x):return sign*(model.functions(x)[1][0]-1)
            def jac(x):model.functions(x);return sign*np.array(model.derivatives(x)[1])
            constraints=[dict(type="eq" if kind=="equality" else "ineq",fun=fun,jac=jac)]
        began=perf_counter();external=minimize(obj,start,jac=grad,bounds=list(zip(lower,upper)),constraints=constraints,method="SLSQP",options=dict(ftol=1e-12,maxiter=100));seconds=perf_counter()-began
        counts=[model.functions_count,model.derivatives_count]
        records.append(dict(case=kind,start=start,solver="scipy-SLSQP",status=str(external.message),seconds=seconds,functions=counts[0],derivatives=counts[1],iterations=int(external.nit),residuals=independent(kind,external.x),target=target))
    for r in records:
        k=r["residuals"]
        r["accepted"]=bool(k and abs(k["objective"]-r["target"])<=1e-6*max(1,abs(r["target"])) and k["primal"]<=1e-7 and k["stationarity"]<=1e-6 and k["complementarity"]<=1e-6 and k["dual_sign"]<=1e-8)
    return dict(scipy=scipy.__version__,numpy=np.__version__,note="Small analytic fixtures, one run each; no performance conclusion. SLSQP residuals use independently reconstructed active multipliers.",runs=records)


if __name__=="__main__":print(json.dumps(run(),indent=2,allow_nan=False))
