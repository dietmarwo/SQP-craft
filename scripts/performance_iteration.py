"""Record native release benchmarks against fixed callbacks and KKT gates."""
import argparse,json,platform,statistics,subprocess
from pathlib import Path
CASES=[('orbit',20),('orbit',40),('separable',128),('restoration',64)]
def measure(binary,repeats=3):
    records=[]
    for run in range(repeats+1):
        order=CASES if run%2==0 else list(reversed(CASES))
        for family,size in order:
            p=subprocess.run([str(Path(binary).resolve()),family,str(size),'reuse'],capture_output=True,text=True,timeout=60)
            if p.returncode:raise RuntimeError(p.stderr)
            row=json.loads(p.stdout);row['run']=run;records.append(row)
    summary=[]
    for family,size in CASES:
        rows=[r for r in records if r['run'] and r['family']==family and r['size']==size]
        metrics={key:statistics.median(r[key] for r in rows) for key in rows[0] if key not in ('family','size','reuse','run')}
        summary.append(dict(family=family,size=size,**metrics))
    return dict(platform=platform.platform(),repeats=repeats,records=records,summary=summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('output');p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    r=measure(a.binary,a.repeats);Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    for row in r['summary']:print(row['family'],row['size'],f"{row['seconds']:.6f}",row['iterations'])
