"""Paired runs of named option sets on the test families.

    python3 scripts/compare_variants.py --variants default,augmented
        [--families stress,mpcc,maratos,baseline] [--qp-solver clarabel|active-set]
        [--json out.json]

Each family is solved once per variant with the same starts. A variant
is a name from VARIANTS, or several joined by "+" (for example
"augmented+al"). For each family and variant the script prints:
- the statuses, total iterations and callbacks (functions plus derivatives);
- the sums of the report counters in COUNTERS;
- how the BFGS matrix was updated over the last three steps of converged
  runs.

It then prints the totals over the problems that every variant solves.
Every convergence claim is checked by the independent first-order evaluator
of examples/compare_baseline.py, and the script fails on a false claim.
For the 13 baseline cases it also reports how many reach the reference
objective.

Families:
- stress: examples/stress_random.py, seeds 1–5, 1,500 problems;
- mpcc: examples/stress_mpcc.py, seeds 1–2, 600 problems;
- maratos: 200 starts of Nocedal & Wright's Example 15.4;
- baseline: the 13 cases of examples/compare_baseline.py.

Further families are added by later steps.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "examples"))
sys.path.insert(0, str(root / "scripts"))
import compare_baseline as cb  # noqa: E402
import compare_corrections as cc  # noqa: E402
import stress_mpcc as sm  # noqa: E402
import stress_random as sr  # noqa: E402

from sqp_craft import SqpOptions, solve  # noqa: E402

VARIANTS: dict[str, dict] = {
    "default": {},
    "augmented": dict(hessian_update="augmented"),
    "soc": dict(second_order_correction=True),
    "elastic": dict(elastic_mode=True),
}

COUNTERS = ["bfgs_damped_updates", "bfgs_modified_updates",
            "hessian_resets", "restoration_steps", "elastic_steps", "second_order_steps",
            "undefined_trials"]


def generated(problem, seeds, count=300):
    for seed in seeds:
        rng = sr.np.random.default_rng(seed)
        for k in range(count):
            spec, model, start = problem(rng)
            yield f"{seed}/{k}", cb.Case("generated", spec, model, start, 0.0, math.inf)


def maratos():
    for k, (spec, model, start) in enumerate(cc.maratos_problems()):
        yield str(k), cb.Case("maratos", spec, model, start, 0.0, math.inf)


def baseline():
    for k, case in enumerate(cb.cases()):
        yield f"{case.name}/{k}", case


FAMILIES = {
    "stress": lambda: generated(sr.problem, range(1, 6)),
    "mpcc": lambda: generated(sm.problem, range(1, 3)),
    "maratos": maratos,
    "baseline": baseline,
}


def options_for(variant: str, qp_solver: str) -> dict:
    settings = dict(max_iterations=300, qp_solver=qp_solver, record_history=True)
    for part in variant.split("+"):
        if part not in VARIANTS:
            sys.exit(f"unknown variant {part!r}; known: {', '.join(VARIANTS)}")
        settings.update(VARIANTS[part])
    return settings


def run_family(name: str, variants: list[str], qp_solver: str, false_claims: list) -> dict:
    cases = list(FAMILIES[name]())
    results = {}
    for variant in variants:
        options = SqpOptions(**options_for(variant, qp_solver))
        records = []
        for key, case in cases:
            r = solve(case.spec, case.make_model(), case.start, method="sqp",
                      method_options=options)
            check = cb.evaluate(case, r.candidate, r.multipliers) if r.candidate else None
            ok = sr.first_order_ok(check)
            if r.status == "converged" and not ok:
                false_claims.append((name, key, variant))
            steps = [h.curvature for h in r.report.history if h.step_length is not None]
            records.append(dict(
                key=key, status=r.status, accepted=ok, iterations=r.iterations,
                callbacks=r.counts.functions + r.counts.derivatives,
                reference=bool(check and check["accepted"]),
                counters={c: getattr(r.report, c, 0) for c in COUNTERS},
                last=steps[-3:] if r.status == "converged" else []))
        results[variant] = records
    return results


def summarize(name: str, results: dict) -> None:
    variants = list(results)
    print(f"\n## {name}\n")
    for variant, records in results.items():
        statuses = collections.Counter(r["status"] for r in records)
        counters = {c: sum(r["counters"][c] for r in records) for c in COUNTERS}
        last = collections.Counter(k for r in records for k in r["last"])
        reference = sum(r["reference"] for r in records)
        print(f"- {variant}: {dict(statuses)}; iterations {sum(r['iterations'] for r in records)}, "
              f"callbacks {sum(r['callbacks'] for r in records)}"
              + (f"; reference objective {reference}/{len(records)}" if name == "baseline" else ""))
        print(f"  counters {', '.join(f'{c} {v}' for c, v in counters.items() if v)}")
        print(f"  last three updates of converged runs: {dict(last)}")
    common = [i for i in range(len(results[variants[0]]))
              if all(results[v][i]["status"] == "converged" for v in variants)]
    totals = ", ".join(
        f"{v} {sum(results[v][i]['iterations'] for i in common)} it / "
        f"{sum(results[v][i]['callbacks'] for i in common)} cb" for v in variants)
    print(f"- converged under every variant: {len(common)}; {totals}")
    flips = [(results[variants[0]][i]["key"], {v: results[v][i]["status"] for v in variants})
             for i in range(len(results[variants[0]]))
             if len({results[v][i]["status"] for v in variants}) > 1]
    if flips:
        print(f"- status changes ({len(flips)}): {flips[:12]}{' ...' if len(flips) > 12 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="default")
    parser.add_argument("--families", default="stress,mpcc,maratos,baseline")
    parser.add_argument("--qp-solver", choices=["clarabel", "active-set"], default="active-set")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    variants = args.variants.split(",")
    false_claims, everything = [], {}
    print(f"QP solver {args.qp_solver}; variants {', '.join(variants)}")
    for name in args.families.split(","):
        everything[name] = run_family(name, variants, args.qp_solver, false_claims)
        summarize(name, everything[name])
    if args.json:
        args.json.write_text(json.dumps(everything, indent=1))
    if false_claims:
        sys.exit(f"converged without passing the independent check: {false_claims}")


if __name__ == "__main__":
    main()
