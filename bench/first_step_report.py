"""Validate retained first-step evidence and regenerate acceptance tables.
Run from the repository root: python -m bench.first_step_report.
"""
from __future__ import annotations
import collections, gzip, json, re, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/benchmarks/first-step'

def read(name):
    p=D/name
    with (gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:
        if '.jsonl' in p.name:return [json.loads(l)for l in f if l.strip()]
        return json.load(f)

def results(name):return [r for r in read(name)if r.get('event')=='result']

def canonical(value,ignore_solver=False):
    ignored={'step_limited_iterations','initial_curvature_scale','callback_calls','actual_functions','actual_derivatives'}
    if ignore_solver:ignored.add('solver')
    if isinstance(value,dict):return {k:canonical(v,ignore_solver)for k,v in value.items()if 'seconds'not in k and k not in ignored}
    if isinstance(value,list):return [canonical(v,ignore_solver)for v in value]
    if isinstance(value,str):return re.sub(r'goldfarb-idnani 0\.(19|20)\.0','goldfarb-idnani VERSION',value)
    return value

def main():
    baseline={r['problem']:r for r in results('baseline-0.19.0.jsonl.gz')}
    rows=results('results.jsonl.gz');assert len(baseline)==1036 and len(rows)==5180
    subsets={r['name']:r['subset']for r in json.loads((ROOT/'docs/benchmarks/selection.json').read_text())['problems']}
    variants=collections.defaultdict(list)
    for r in rows:variants[r['solver']].append(r)
    false=sum(bool(r['claimed'])and not r['check']['accepted']for r in rows)
    cb_base={r['problem']:r for r in results('callbacks/baseline.jsonl.gz')}
    callback_selection=read('callbacks/selection.json')
    stats={};complete=len(cb_base)==len(callback_selection['baseline'])
    for name,rr in variants.items():
        assert len(rr)==1036 and len({r['problem']for r in rr})==1036
        gains=[r['problem']for r in rr if r['check']['accepted']and not baseline[r['problem']]['check']['accepted']]
        losses=[r['problem']for r in rr if not r['check']['accepted']and baseline[r['problem']]['check']['accepted']]
        delta={s:sum(int(r['check']['accepted'])-int(baseline[r['problem']]['check']['accepted'])for r in rr if subsets[r['problem']]==s)for s in ['unconstrained','bounds','linear','nonlinear']}
        record=dict(accepted=sum(bool(r['check']['accepted'])for r in rr),net=len(gains)-len(losses),gains=gains,losses=losses,subset_delta=delta,false_claims=sum(bool(r['claimed'])and not r['check']['accepted']for r in rr))
        if name.startswith('auto+'):
            untriggered=[r for r in rr if not r['report'].get('step_limited_iterations') and r['report'].get('initial_curvature_scale')is None]
            mismatch=[r['problem']for r in untriggered if baseline[r['problem']]['status']!='error' and canonical(r,True)!=canonical(baseline[r['problem']],True)]
            record.update(untriggered=len(untriggered),untriggered_nonerror_mismatches=mismatch)
        if name!='auto':
            cb={r['problem']:r for r in results('callbacks/'+name+'.jsonl.gz')}
            complete &= len(cb)==len(callback_selection[name])
            paired=[(cb_base[p]['callback_calls'],cb[p]['callback_calls'])for p in callback_selection[name]if p in cb and p in cb_base]
            ratios=[b/a for a,b in paired]
            record.update(callback_pairs=len(paired),median_callback_ratio=statistics.median(ratios),baseline_median_callbacks=statistics.median(a for a,b in paired),variant_median_callbacks=statistics.median(b for a,b in paired))
            record['median_callback_count_ratio']=record['variant_median_callbacks']/record['baseline_median_callbacks']
            record['default_gate']=(name.startswith('auto+') and record['net']>=10 and min(delta.values())>=0 and record['false_claims']==0 and record['median_callback_ratio']<=1.02 and record['median_callback_count_ratio']<=1.02 and not record.get('untriggered_nonerror_mismatches',[]))
        stats[name]=record
    # Failure conversion and fixed non-timing records.
    default={r['problem']:r for r in variants['auto']}
    differences=[dict(problem=p,before=a['status'],after=default[p]['status'])for p,a in baseline.items()if canonical(a)!=canonical(default[p])]
    comparable=[p for p,a in baseline.items()if a['status']not in ['error','time_limit','timeout']and default[p]['status']not in ['time_limit','timeout']]
    identical=sum(canonical(baseline[p])==canonical(default[p])for p in comparable)
    panel0={(r['problem'],r['solver']):r for r in results('panel-baseline.jsonl.gz')};panel1={(r['problem'],r['solver']):r for r in results('panel-final.jsonl.gz')}
    panel_mismatch=[list(k)for k in panel0 if canonical(panel0[k])!=canonical(panel1[k])]
    stress0=read('stress-baseline.jsonl.gz');stress1=read('stress-final.jsonl.gz');stress_on=read('stress-safeguard.jsonl.gz')
    stress_mismatch=sum(canonical(a)!=canonical(b)for a,b in zip(stress0,stress1))
    stress_false=sum(r['status']=='converged'and not r['accepted']for r in stress_on)
    stress_losses=sum(a['status']=='converged'and b['status']!='converged'for a,b in zip(stress0,stress_on))
    status0=results('status-baseline.jsonl.gz');status1=results('status-final.jsonl.gz')
    standard0=[r for r in status0 if r['solver']!='active-set+augmented'];standard1=[r for r in status1 if r['solver']!='active-set+augmented']
    status=dict(standard_runs=len(standard1),baseline_errors=sum(r['status']=='error'for r in standard0),final_errors=sum(r['status']=='error'for r in standard1),extended_runs=len(status1),extended_baseline_errors=sum(r['status']=='error'for r in status0),extended_final_errors=sum(r['status']=='error'for r in status1),false_claims=sum(r['claimed']and not r['check']['accepted']for r in status1))
    summary=dict(baseline_accepted=sum(bool(r['check']['accepted'])for r in baseline.values()),variants=stats,false_claims=false,default='major_step_limit=None, initial_curvature_scaling=False',decision='No variant passes the no-subset-loss gate; retain the legacy default step policy.',default_comparable=len(comparable),default_identical=identical,default_differences=differences,panel_runs=len(panel1),panel_mismatches=panel_mismatch,stress_runs=len(stress1),stress_mismatches=stress_mismatch,enabled_stress_false_claims=stress_false,enabled_stress_lost_convergences=stress_losses,status_panel=status,callbacks_complete=complete,tutorials_equal=all(r['equal_without_timings']for r in read('tutorial-comparison.json.gz')))
    (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# First-step acceptance results','','Fresh 0.19.0 automatic baseline: **814 / 1,036 accepted**. Accepted means the unchanged independent first-order check accepts the returned point. No variant makes a false convergence claim.','','| Variant | Accepted | Net | Unconstrained | Bounds | Linear | Nonlinear | Median paired actual callback ratio |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in stats.items():
        d=r['subset_delta'];ratio=f"{r['median_callback_ratio']:.4f}"if name!='auto'else'1.0000'
        lines.append(f"| {name} | {r['accepted']} | {r['net']:+d} | {d['unconstrained']:+d} | {d['bounds']:+d} | {d['linear']:+d} | {d['nonlinear']:+d} | {ratio} |")
    lines+=['','## Decision','','Keep `major_step_limit=None` and `initial_curvature_scaling=False`. The delta-2 proposed safeguard gains 23 and loses 13, meeting +10 net but losing one net bound-constrained problem. The delta-10 and step-only variants also lose in that subset. The explicit active-set variant is compared with the automatic baseline as an additional policy comparison; it is not a replacement automatic default.','','## Compatibility and validation','',f"- {identical}/{len(comparable)} uncensored, previously non-error automatic records match 0.19.0 exactly after timings and new disabled diagnostics are removed. HS25NE's former exception becomes a status. The two wall-limited records retain their status with timing-dependent work counts.",f'- Backend panel: {len(panel1)} records, {len(panel_mismatch)} non-timing differences.',f'- Default stress: {len(stress1)} records, {stress_mismatch} non-timing differences after version normalization.',f'- Enabled safeguard stress: {len(stress_on)} runs, {stress_false} false convergence claims, {stress_losses} lost convergences.',f"- Status panel: {status['baseline_errors']} exceptions in the planned {status['standard_runs']} baseline runs, {status['final_errors']} after the fix. The augmented-update supplement makes {status['extended_runs']} records with {status['extended_baseline_errors']} baseline exceptions and {status['extended_final_errors']} final exceptions.",'- SQP1–SQP6 and X2 numerical tutorial summaries match; orbit/Maratos and job fixtures are covered by the Rust/Python suites.','','## Callbacks','','The full harness counts distinct model points. `callbacks/` replays commonly accepted pairs and records actual `SolveResult.counts` function and derivative invocations, excluding independent post-solve validation. Both the median per-problem new/old actual-call ratio and the ratio of actual-call medians are checked against 1.02. No tolerance was changed. All failed/slower full-benchmark variants and the rejected early-guard experiment are retained.','']
    (D/'tables.md').write_text('\n'.join(lines))
    assert complete and false==0 and identical==len(comparable) and not panel_mismatch and stress_mismatch==0 and stress_false==0 and stress_losses==0 and status['final_errors']==0 and summary['tutorials_equal'],summary
    assert not any(r.get('default_gate')for r in stats.values())
    print(json.dumps({k:summary[k]for k in ['default','decision','default_identical','panel_runs','stress_runs','status_panel','callbacks_complete']},indent=2))
if __name__=='__main__':main()
