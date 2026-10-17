"""Explicit curvature options retain original KKT and callback contracts."""
import pytest
from sqp_craft import SqpOptions,solve,kkt_residuals
from test_sqp import Quadratic,free

@pytest.mark.parametrize('block',[0,1,2,8,256])
@pytest.mark.parametrize('reuse',[False,True])
def test_explicit_blocks_known_quadratic(block,reuse):
 spec=free();r=solve(spec,Quadratic(),[0.,0.],method='sqp',method_options=SqpOptions(hessian_block_size=block,qp_reuse_workspace=reuse))
 assert r.status=='converged' and r.evaluation.objective<1e-10
 k=kkt_residuals(spec,r.candidate,r.evaluation,r.multipliers)
 assert k.primal<=1e-7 and k.stationarity<=1e-6 and k.complementarity<=1e-6 and k.dual_sign<=1e-8

def test_invalid_block_size_rejected():
 with pytest.raises(ValueError):solve(free(),Quadratic(),[0.,0.],method='sqp',method_options=SqpOptions(hessian_block_size=257))


def test_small_hessian_cancellation_regression():
 import json,math
 from pathlib import Path
 import numpy as np
 from sqp_craft import ProblemSpec
 data=json.loads((Path(__file__).resolve().parents[1]/'docs/performance/regression-seed4-case86.json').read_text())
 target=np.asarray(data['target']);rows=[(np.asarray(r['a']),r['s'],np.asarray(r['y'])) for r in data['rows']]
 class Model:
  def functions(self,x):
   x=np.asarray(x);return float(np.sum((x-target)**2)),[float(a@x+s*np.sum((x-y)**2)) for a,s,y in rows]
  def derivatives(self,x):
   x=np.asarray(x);return list(2*(x-target)),list(np.concatenate([a+2*s*(x-y) for a,s,y in rows]))
 n=len(target);m=len(rows)
 spec=ProblemSpec(lower=data['lower'],upper=data['upper'],constraint_lower=[-math.inf if v=='-inf' else v for v in data['row_lower']],constraint_upper=data['row_upper'],jacobian_indptr=[i*n for i in range(m+1)],jacobian_indices=list(range(n))*m)
 result=solve(spec,Model(),data['start'],method='sqp',method_options=SqpOptions(max_iterations=300,qp_reuse_workspace=True))
 assert result.status=='converged',result.message
 k=kkt_residuals(spec,result.candidate,result.evaluation,result.multipliers)
 assert k.primal<=1e-7 and k.stationarity<=1e-6 and k.complementarity<=1e-6 and k.dual_sign<=1e-8
