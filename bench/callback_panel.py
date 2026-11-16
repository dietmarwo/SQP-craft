"""Re-evaluate commonly accepted pairs to measure actual callback invocations.

python -m bench.callback_panel baseline.jsonl results.jsonl /absolute/new/output \
  --baseline-python /absolute/baseline/.venv/bin/python

The full benchmark's functions/derivatives count distinct model points.
This secondary panel records SolveResult.counts to measure actual calls.
Both interpreters run the current harness, with their own installed packages.
"""
import argparse, concurrent.futures, gzip, json, os, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(path):
 opener=gzip.open if path.suffix=='.gz' else open
 with opener(path,'rt')as f:return [r for r in map(json.loads,f)if r.get('event')=='result']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=pathlib.Path);p.add_argument('results',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);p.add_argument('--baseline-python',required=True);a=p.parse_args()
 old={r['problem']:r for r in read(a.baseline)};rows=read(a.results)
 variants=['auto+step','auto+safeguard','auto+safeguard10','active-set+safeguard']
 sets={v:sorted(r['problem']for r in rows if r['solver']==v and r['check']['accepted'] and old[r['problem']]['check']['accepted'])for v in variants}
 sets['baseline']=sorted(set().union(*map(set,sets.values())))
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'selection.json').write_text(json.dumps(sets,indent=2)+'\n')
 def run(v):
  interpreter=a.baseline_python if v=='baseline'else sys.executable
  solver='auto'if v=='baseline'else v
  cmd=[interpreter,'-m','bench.run','docs/benchmarks/selection.json',str((a.output/(v+'.jsonl')).resolve()),'--solvers',solver,'--problems',','.join(sets[v]),'--jobs','2']
  with (a.output/(v+'.log')).open('w')as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=log,check=True,timeout=1200,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:list(pool.map(run,['baseline',*variants]))
if __name__=='__main__':main()
