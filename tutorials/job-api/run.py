#!/usr/bin/env python3
"""Runnable SQP job tutorial. Build the release binaries first:

    cargo build --release --locked -p optimization-runner -p sqp-job-worker
"""
import argparse, json, math, os, subprocess, sys, tempfile
from pathlib import Path
try:
    import jsonschema
except ImportError:
    jsonschema=None
REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
WORKER='sqp'
BINARY='sqp-job-worker'
EXAMPLES=['simplex', 'rosenbrock', 'quadratic-clarabel', 'simplex-evaluate']
sys.path.insert(0,str(REPO/'scripts/job-api'))
from shared_root import shared_root
# The schemas live in optimization-jobs, at the revision Cargo.toml pins.
SCHEMAS=shared_root()/'docs/job-api/schemas'
if jsonschema is None and os.environ.get('JOB_API_REQUIRE_SCHEMA')=='1':
    raise SystemExit('the jsonschema package is required (JOB_API_REQUIRE_SCHEMA=1)')

def conforms(schema,value):
    """Checks value against a published JSON Schema if jsonschema is installed."""
    if jsonschema is not None:
        jsonschema.validate(value,json.loads((SCHEMAS/schema).read_text()),cls=jsonschema.Draft202012Validator)

p=argparse.ArgumentParser();p.add_argument('--root',type=Path);args=p.parse_args()
root=(args.root or Path(tempfile.mkdtemp(prefix=WORKER+'-jobs-'))).resolve()
runner=REPO/'target/release/optimization-run'
def call(*args):
    run=subprocess.run([str(runner),'--root',str(root),*map(str,args)],capture_output=True,text=True,check=True)
    return json.loads(run.stdout)
if not (root/'node.json').exists():call('node','init','--capacity',8)
call('workers','add','--id',WORKER,'--executable',REPO/'target/release'/BINARY)
report=[]
for example in EXAMPLES:
    spec=HERE/(example+'.json');payload=json.loads(spec.read_text())['payload']
    conforms('optimization.job-0.1.schema.json',json.loads(spec.read_text()))
    call('validate','--spec',spec)
    job=call('run','--spec',spec);result=call('result',job['directory'])
    conforms('optimization.result-0.2.schema.json',result)
    assert job['state']=='completed',result
    assert result['runner']['accounting_complete'],result
    for record in result['results']:
        outcome=record['outcome'];evidence=outcome['evidence']
        if example=='simplex-evaluate':
            # sum(x) = 1 holds for the first two points only.
            x=payload['points'][record['task_id']];feasible=math.isclose(sum(x),1.)
            assert outcome['termination']=='evaluated' and outcome['feasible']==feasible,outcome
            assert outcome['claim']==('feasible' if feasible else 'none'),outcome
            assert math.isclose(outcome['objective'],sum(v*v for v in x)),outcome
        else:
            # Converged, and the KKT residuals recomputed outside the solver pass.
            assert outcome['claim']=='first_order',outcome
            assert evidence=={'level':'rechecked','checks':['objective','bounds','constraints','kkt']},outcome
            assert math.isclose(outcome['objective'],.25 if example=='simplex' else 0.,abs_tol=1e-6),outcome
    report.append({'example':example,'job_id':job['id'],'directory':job['directory'],'state':job['state'],'completed_tasks':result['completed_tasks'],'first_outcome':result['results'][0]['outcome'],'work':result['work']})
output={'node':str(root),'worker':WORKER,'runs':report}
(root/'tutorial-report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
