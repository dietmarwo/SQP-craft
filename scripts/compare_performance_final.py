"""Alternating baseline/final native processes; no timing assertions."""
import argparse,hashlib,json,platform,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('output');p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
assert a.repeats>0
cases=[('orbit',20),('orbit',40),('separable',128),('restoration',64)]
variants=[('baseline-reuse',a.baseline,True,False),('final-reuse',a.candidate,True,False),('final-blocks',a.candidate,True,True),('baseline-cold',a.baseline,False,False),('final-cold',a.candidate,False,False)]
cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
r=dict(cpu=cpu,platform=platform.platform(),repeats=a.repeats,warmups=1,method='one fresh process per call; rotating and reversed order; one numerical thread; generation excluded; no CPU affinity/exclusive-host guarantee',binary_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('baseline',a.baseline),('candidate',a.candidate)]},records=[])
output=Path(a.output)
jobs=[(f,n,*v) for f,n in cases for v in variants]
for run in range(a.repeats+1):
 order=jobs[run%len(jobs):]+jobs[:run%len(jobs)]
 if run%2:order=list(reversed(order))
 for f,n,label,binary,reuse,blocked in order:
  block=({'orbit':4,'separable':2,'restoration':1}[f] if blocked else 0)
  c=subprocess.run([str(Path(binary).resolve()),f,str(n),'reuse' if reuse else 'cold',str(block)],capture_output=True,text=True,timeout=60)
  if c.returncode:raise RuntimeError((label,f,n,c.stderr))
  row=json.loads(c.stdout);row.update(variant=label,run=run);r['records'].append(row)
  output.write_text(json.dumps(r,indent=2)+'\n')
 print('completed comparison round',run,flush=True)
r['summary']=[]
for f,n in cases:
 for label,*_ in variants:
  rows=[e for e in r['records'] if e['run']>0 and e['family']==f and e['size']==n and e['variant']==label]
  keys=['seconds','iterations','qp_calls','qp_iterations','bfgs_seconds','conversion_seconds','engine_seconds','setups','updates','objective','primal','stationarity']
  r['summary'].append(dict(family=f,size=n,variant=label,**{k:statistics.median(e[k] for e in rows) for k in keys},block_hessian_fallbacks=statistics.median(e.get('block_hessian_fallbacks',0) for e in rows)))
output.write_text(json.dumps(r,indent=2)+'\n')
