"""Explicit Hessian-approximation experiments; record failed orbit trials."""
import json,subprocess,sys
from pathlib import Path
binary=str(Path(sys.argv[1]).resolve());records=[]
for family,size,block in [('separable',128,2),('separable',128,8),('restoration',64,1),('restoration',64,8),('orbit',20,4),('orbit',40,4)]:
 p=subprocess.run([binary,family,str(size),'reuse',str(block)],capture_output=True,text=True,timeout=60)
 if p.returncode:r=dict(family=family,size=size,hessian_block_size=block,accepted=False,error=p.stderr)
 else:r=dict(json.loads(p.stdout),accepted=True)
 records.append(r)
Path(sys.argv[2]).write_text(json.dumps(records,indent=2)+'\n')
assert all(r['accepted'] for r in records if r['family']!='orbit'),records
for r in records:print(r['family'],r['size'],r['hessian_block_size'],r['accepted'],r.get('seconds'))
