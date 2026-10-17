"""Same random inputs and independent NumPy KKT evaluator across releases."""
import argparse,collections,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
import stress_random as sr
import compare_baseline as cb
from sqp_craft import SqpOptions,solve,__version__
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--count',type=int,default=300);p.add_argument('--seed',type=int,default=1);a=p.parse_args()
rng=sr.np.random.default_rng(a.seed);records=[];counts=collections.Counter();false_claims=[];errors=[]
for case_id in range(a.count):
 spec,model,start=sr.problem(rng);case=cb.Case('random',spec,model,start,0.,math.inf)
 for mode,reuse,rest in [('cold',False,10),('reuse',True,10),('restoration_off',True,0)]:
  try:
   r=solve(spec,model(),start,method='sqp',method_options=SqpOptions(max_iterations=300,qp_reuse_workspace=reuse,max_restoration_steps=rest))
   check=cb.evaluate(case,r.candidate,r.multipliers) if r.candidate is not None else None
   ok=sr.first_order_ok(check)
   if r.status=='converged' and (not ok or check['certificate']!='solver'):false_claims.append([case_id,mode])
   counts[(mode,r.status,ok)]+=1
   records.append(dict(case=case_id,mode=mode,status=r.status,accepted=ok,check=check,iterations=r.iterations))
  except Exception as e:
   errors.append(dict(case=case_id,mode=mode,error=repr(e)))
result=dict(version=__version__,seed=a.seed,cases=a.count,solver_calls=len(records)+len(errors),false_claims=false_claims,errors=errors,counts=[dict(mode=k[0],status=k[1],accepted=k[2],count=v) for k,v in sorted(counts.items())],records=records)
Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(__version__,a.seed,len(records),'errors',len(errors),'false_claims',len(false_claims))
assert not errors and not false_claims
