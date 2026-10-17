"""Run one checkpoint's checks, release measurements and local Git tag."""
import argparse,json,os,re,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('iteration',type=int);a=p.parse_args();i=a.iteration
repo=Path.cwd();val=repo.parents[1]/'validation/sqp-performance';val.mkdir(exist_ok=True)
version=f'0.{4+i}.0'
for name in ['Cargo.toml','pyproject.toml','python/sqp_craft/__init__.py']:
 f=Path(name);s=f.read_text();s=re.sub(r'(version\s*=\s*|__version__\s*=\s*)"0\.\d+\.0"',lambda m:m.group(1)+'"'+version+'"',s);f.write_text(s)
def run(cmd,name):
 with (val/f'{i:02}-{name}.log').open('w') as log:
  r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{name}: {(val/f"{i:02}-{name}.log").read_text()[-5000:]}')
run(['cargo','check','--workspace','--offline'],'check')
run(['cargo','fmt','--all'],'fmt')
run(['cargo','clippy','--workspace','--all-targets','--locked','--','-D','warnings'],'clippy')
run(['cargo','test','--workspace','--release','--locked'],'tests')
run(['cargo','build','--release','--locked','-p','sqp-craft-solver','--example','performance'],'build')
shutil.copy2('target/release/examples/performance',val/f'bench-{i:02}')
if i>=9:run(['python3','scripts/benchmark_structured.py','target/release/examples/performance',f'docs/performance/iteration-{i:02}-blocks.json'],'blocks')
run(['python3','scripts/performance_iteration.py','target/release/examples/performance',f'docs/performance/iteration-{i:02}.json'],'bench')
current=json.loads(Path(f'docs/performance/iteration-{i:02}.json').read_text());prior=json.loads(Path(f'docs/performance/iteration-{i-1:02}.json').read_text())
lines=[f'# Performance iteration {i:02} results', '',f'Version {version}. Full workspace release tests and Clippy passed. Each native benchmark independently checks original-unit KKT residuals; generated problems also have known objective zero.', '', '| Case | Previous ms | Current ms | Previous/current ratio | Outer steps |','| --- | ---: | ---: | ---: | ---: |']
for old,new in zip(prior['summary'],current['summary']):
 assert (old['family'],old['size'])==(new['family'],new['size'])
 lines.append(f"| {new['family']} {new['size']} | {1000*old['seconds']:.3f} | {1000*new['seconds']:.3f} | {old['seconds']/new['seconds']:.3f} | {new['iterations']} |")
lines+=['','Three measured samples after one warmup, same fixed callbacks, one numerical thread, workspace reuse enabled. These sequential medians include host variation; the final rotating comparison is the stronger timing evidence. Ratios below one are disclosed regressions/measurement variation, not improvements. See the raw JSON for phase times, iteration counts and residuals.','']
Path(f'docs/performance/iteration-{i:02}-results.md').write_text('\n'.join(lines))
logs=Path('docs/performance/validation');logs.mkdir(exist_ok=True)
for name in ['tests','clippy']:shutil.copy2(val/f'{i:02}-{name}.log',logs/f'{i:02}-{name}.log')
subprocess.run(['git','add','-A'],check=True)
subprocess.run(['git','commit','-m',f'Performance iteration {i:02}: '+Path(f'docs/performance/iteration-{i:02}-plan.md').read_text().splitlines()[0].lstrip('# ')],check=True)
subprocess.run(['git','tag',f'sqp-performance-{i:02}'],check=True)
for new in current['summary']:print(new['family'],new['size'],round(new['seconds']*1000,3),new['iterations'])
