#!/usr/bin/env python3
"""Runnable SQP job tutorial. Build the release binaries and the custom_worker example first:

    cargo build --release --locked -p optimization-runner -p sqp-job-worker
    cargo build --release --locked -p sqp-job-worker --example custom_worker
"""
import argparse, hashlib, json, math, os, subprocess, sys, tempfile
from pathlib import Path
try:
    import jsonschema
except ImportError:
    jsonschema=None
REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
WORKER='sqp'
BINARY='sqp-job-worker'
# An application worker with a data-backed model of its own (examples/custom_worker.rs).
APPLICATION=('sqp-curve-fit','examples/custom_worker')
EXAMPLES=['simplex', 'rosenbrock', 'quadratic-clarabel', 'simplex-evaluate', 'decay-fit']
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

def payload_conforms(description,spec):
    """Checks a job's payload against the payload schema its worker describes."""
    schema=description['payload_schemas'][spec['payload_schema']]
    if jsonschema is not None:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(spec['payload'],schema,cls=jsonschema.Draft202012Validator)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,help='node directory (default: a new temporary directory)')
p.add_argument('--bin-dir',type=Path,default=Path(os.environ.get('CARGO_TARGET_DIR',REPO/'target'))/'release',
               help='directory with optimization-run, '+BINARY+' and '+APPLICATION[1]+' (default: $CARGO_TARGET_DIR/release or target/release)')
args=p.parse_args()
root=(args.root or Path(tempfile.mkdtemp(prefix=WORKER+'-jobs-'))).resolve()
bin_dir=args.bin_dir.resolve()
runner,worker,application=bin_dir/'optimization-run',bin_dir/BINARY,bin_dir/APPLICATION[1]
for binary in [runner,worker,application]:
    if not binary.is_file():p.error(f'{binary} not found; build the release binaries and the custom_worker example first')
def call(*args):
    run=subprocess.run([str(runner),'--root',str(root),*map(str,args)],capture_output=True,text=True,check=True)
    return json.loads(run.stdout)

def check_fit(spec,results):
    """Rechecks the decay fit from decay.csv, outside the worker: the loss, the
    anchor row, and stationarity (bounds inactive, so the gradient of the loss
    must be a multiple of the anchor row's gradient)."""
    rows=[tuple(map(float,line.split(','))) for line in (HERE/'decay.csv').read_text().split()[1:]]
    anchor=spec['payload']['config']['anchor_time'];y_anchor=next(y for t,y in rows if t==anchor)
    objectives=[]
    for record in results:
        outcome=record['outcome'];a,b,c=outcome['x']
        loss=.5*sum((a*math.exp(-b*t)+c-y)**2 for t,y in rows)
        assert math.isclose(outcome['objective'],loss,rel_tol=1e-12),(outcome['objective'],loss)
        assert abs(a*math.exp(-b*anchor)+c-y_anchor)<=1e-7,outcome['x']
        g=[0.,0.,0.]
        for t,y in rows:
            e=math.exp(-b*t);r=a*e+c-y;g[0]+=r*e;g[1]-=r*a*t*e;g[2]+=r
        e=math.exp(-b*anchor);h=[e,-a*anchor*e,1.]
        lam=sum(u*v for u,v in zip(g,h))/sum(v*v for v in h)
        assert max(abs(u-lam*v) for u,v in zip(g,h))<=1e-5,(g,h)
        # decay.csv is 3*exp(-0.7*t)+0.5 plus noise of standard deviation 0.01.
        assert abs(a-3)<.05 and abs(b-.7)<.03 and abs(c-.5)<.03,outcome['x']
        assert outcome['details']['options']['elastic_mode'] is True,outcome['details']
        objectives.append(outcome['objective'])
    # The same point from every start.
    assert max(objectives)-min(objectives)<=1e-9,objectives

def schemas_reject(description,spec,mutations):
    """The payload schema rejects what the worker rejects: each mutation of a valid payload."""
    if jsonschema is None:return
    validator=jsonschema.Draft202012Validator(description['payload_schemas'][spec['payload_schema']])
    directory=root/'rejected';directory.mkdir(exist_ok=True);path=directory/'job.json'
    for item in spec['inputs']:(directory/item['file']).write_bytes((HERE/item['file']).read_bytes())
    def valid(payload):
        path.write_text(json.dumps({**spec,'payload':payload}))
        return subprocess.run([str(runner),'--root',str(root),'validate','--spec',str(path)],capture_output=True).returncode==0
    assert validator.is_valid(spec['payload']) and valid(spec['payload'])
    for change in mutations:
        bad={**spec['payload'],**change}
        assert not validator.is_valid(bad) and not valid(bad),change

def run_examples(worker_id,description):
    """Turns every example of a worker's describe into a job, as an MCP-only
    client would, and runs it."""
    runs=[]
    for i,example in enumerate(description['examples']):
        directory=root/'describe-examples'/f'{worker_id}-{i}';directory.mkdir(parents=True,exist_ok=True)
        inputs=[]
        for item in example['inputs']:
            data=item['content_text'].encode();(directory/item['file']).write_bytes(data)
            inputs.append({'name':item['name'],'file':item['file'],'sha256':hashlib.sha256(data).hexdigest()})
        spec={'schema_version':'optimization.job/0.1','request_key':f'tutorial.describe.{worker_id}.{i}','worker':worker_id,
              'payload_schema':example['profile'],'payload':example['payload'],'resources':example['resources'],
              'limits':example['limits'],'inputs':inputs}
        conforms('optimization.job-0.1.schema.json',spec);payload_conforms(description,spec)
        path=directory/'job.json';path.write_text(json.dumps(spec,indent=2)+'\n')
        assert call('validate','--spec',path)['resolved']['payload']==example['payload'],example
        job=call('run','--spec',path);result=call('result',job['directory'])
        conforms('optimization.result-0.2.schema.json',result)
        assert job['state']=='completed' and result['completed_tasks']>=1,result
        outcome=result['results'][0]['outcome']
        assert outcome['claim']==('first_order' if example['profile']=='sqp.solve/1' else 'feasible' if outcome['feasible'] else 'none'),outcome
        runs.append({'worker':worker_id,'profile':example['profile'],'problem':example['payload']['problem'],'claim':outcome['claim']})
    return runs

if not (root/'node.json').exists():call('node','init','--capacity',8)
call('workers','add','--id',WORKER,'--executable',worker)
call('workers','add','--id',APPLICATION[0],'--executable',application)
descriptions={w:call('problems','describe',w) for w in (WORKER,APPLICATION[0])}
for w,d in descriptions.items():
    # Discovery: a schema and an example for every profile.
    assert set(d['payload_schemas'])==set(d['profiles']),w
    assert {e['profile'] for e in d['examples']}==set(d['profiles']),w
report=[]
for example in EXAMPLES:
    spec_path=HERE/(example+'.json');spec=json.loads(spec_path.read_text());payload=spec['payload']
    conforms('optimization.job-0.1.schema.json',spec);payload_conforms(descriptions[spec['worker']],spec)
    call('validate','--spec',spec_path)
    job=call('run','--spec',spec_path);result=call('result',job['directory'])
    conforms('optimization.result-0.2.schema.json',result)
    assert job['state']=='completed',result
    assert result['runner']['accounting_complete'],result
    if example=='decay-fit':check_fit(spec,result['results'])
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
            if example!='decay-fit':
                assert math.isclose(outcome['objective'],.25 if example=='simplex' else 0.,abs_tol=1e-6),outcome
    report.append({'example':example,'worker':spec['worker'],'job_id':job['id'],'directory':job['directory'],'state':job['state'],'completed_tasks':result['completed_tasks'],'first_outcome':result['results'][0]['outcome'],'work':result['work']})
simplex=json.loads((HERE/'simplex.json').read_text());fit=json.loads((HERE/'decay-fit.json').read_text())
schemas_reject(descriptions[WORKER],simplex,[{'ignored_option':True},{'backend':'clarabel','elastic_mode':True},
    {'problem_version':'9'},{'problem':'benchmark.unknown'},{'problem':'benchmark.rosenbrock','dimension':1},
    {'config':{'scale':1}},{'max_callback_calls':0},{'feasibility_tolerance':0.}])
schemas_reject(descriptions[APPLICATION[0]],fit,[{'config':{'anchor_time':'t'}},{'config':{'anchor':1.}},{'dimension':2}])
described=[run for w,d in descriptions.items() for run in run_examples(w,d)]
output={'node':str(root),'worker':WORKER,'application_worker':APPLICATION[0],'runs':report,'describe_examples':described}
(root/'tutorial-report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
