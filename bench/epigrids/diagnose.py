"""Capture the rejected reference-start QP without changing any tolerance."""
import argparse,hashlib,json,pathlib
import numpy as np
import run as p

ap=argparse.ArgumentParser();ap.add_argument('--case',type=pathlib.Path,required=True);ap.add_argument('--reference',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
d,reference,identity=p.load(a.case);g=p.Grid(d);rr=json.loads(a.reference.read_text());rdata={'bus':np.array(rr['bus']),'gen':np.array(rr['gen'])};start=g.x_from(rdata)
settings=p.sqp.SqpOptions(hessian='partitioned',qp_solver='clarabel',max_iterations=300,max_callback_calls=5000,time_limit_seconds=60,stationarity_tolerance=1e-10,complementarity_tolerance=1e-10,qp_tolerance=1e-10,qp_reuse_workspace=True,record_qp_limit=16)
r=p.sqp.solve(g.spec(),g,start,method='sqp',method_options=settings)
trace=r.report.qp_trace
record={'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'settings':settings,'identity':identity,'native_status':r.status,'message':r.message,'steps':r.iterations,'trace_count':len(trace),'qp_status':r.report.qp_status,'candidate':r.candidate,'last_qp':trace[-1] if trace else None,'null_bounds':'lower null means -infinity; upper null means +infinity; non-bound null is unknown'}
p.save(a.output/'rejected-qp.json',record)
print(r.status,r.report.qp_status,len(trace))
