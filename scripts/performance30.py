"""Paired, pinned, fixed-work measurements for the thirty-iteration series.

performance30.py OUTPUT previous=PERFORMANCE,KERNELS candidate=PERFORMANCE,KERNELS
Build/test first. Processes are serial; one warmup and three measured samples.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, statistics, subprocess
from pathlib import Path

NLP = [('orbit',20),('orbit',40),('separable',128),('restoration',64)]
KERNELS = [('validate',128,1000),('hessian',128,500),('csr',128,5000),
           ('csr-dense',128,1000),('transpose',128,5000),('kkt',128,1000),
           ('bfgs',128,200),('cholesky',96,100),('inverse',96,100),
           ('qp-dense',64,30),('qp-pairs',64,30),('qp-free',128,30),
           ('qp-box',128,30),('qp-zero',64,30),('qp-free-rows',128,30),
           ('qp-fixed',64,30),('qp-dense-normals',64,30)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('builds',nargs='+');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--kernels-only',action='store_true');a=p.parse_args()
    assert a.repeats>=1
    builds={label:tuple(map(lambda s:str(Path(s).resolve()),paths.split(','))) for label,paths in (v.split('=',1)for v in a.builds)}
    assert all(len(paths)==2 for paths in builds.values())
    jobs=[('kernel',family,size,count,label) for family,size,count in KERNELS for label in builds]
    if not a.kernels_only:jobs += [('nlp',family,size,1,label)for family,size in NLP for label in builds]
    cpu=min(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity')else None
    def pin():
        if cpu is not None:os.sched_setaffinity(0,{cpu})
    result=dict(platform=platform.platform(),cpu_affinity=cpu,repeats=a.repeats,warmups=1,
                binary_sha256={label:[hashlib.sha256(Path(v).read_bytes()).hexdigest()for v in paths]for label,paths in builds.items()},records=[])
    for run in range(a.repeats+1):
        shift=run%len(jobs);order=jobs[shift:]+jobs[:shift]
        if run%2:order=order[::-1]
        for kind,family,size,count,label in order:
            binary=builds[label][kind=='kernel']
            command=[binary,family,str(size)]+([str(count)]if kind=='kernel'else['reuse','0','active-set'])
            r=subprocess.run(command,capture_output=True,text=True,timeout=120,preexec_fn=pin if cpu is not None else None)
            if r.returncode:raise RuntimeError(f'{label} {kind} {family}: {r.stderr[-4000:]}')
            q=json.loads(r.stdout);assert q['seconds']>0
            result['records'].append(dict(q,kind=kind,build=label,run=run))
    result['summary']=[]
    for kind,family,size,count,_ in jobs:
        if any(r['kind']==kind and r['family']==family and r['size']==size for r in result['summary']):continue
        row=dict(kind=kind,family=family,size=size)
        for label in builds:
            selected=[r for r in result['records']if r['run']and r['kind']==kind and r['family']==family and r['size']==size and r['build']==label]
            row[label]=statistics.median(r['seconds']for r in selected)
            if kind=='nlp':row[label+'_iterations']=statistics.median(r['iterations']for r in selected)
        labels=list(builds)
        if len(labels)==2:row['speedup']=row[labels[0]]/row[labels[1]]
        result['summary'].append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    for row in result['summary']:
        print(row['kind'],row['family'],row['size'],' '.join(f'{label}={row[label]*1000:.4f}ms'for label in builds),'ratio',round(row.get('speedup',0),4),flush=True)

if __name__=='__main__':main()
