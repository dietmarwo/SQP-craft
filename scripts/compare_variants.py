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
- baseline: the 13 cases of examples/compare_baseline.py;
- linear: examples/stress_linear.py, seeds 1–2, 600 problems whose model is
  undefined outside linear rows;
- stress-scaled, linear-scaled: the stress and linear problems of seeds 1–2
  in badly scaled units, x = S x′ and c′ = T c with S and T random powers of
  ten in [10⁻², 10²];
- orbit-scaled: 20 such copies of each orbit transfer (N = 20 and 40).

The variant "linear" declares the rows each family knows to be linear:
- the sum row of the mpcc family;
- the duration rows of the linear family;
- x₀ + x₁ = 1 and u(t_f) = 0 in the baseline cases.

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
import stress_linear as sl  # noqa: E402
import stress_mpcc as sm  # noqa: E402
import stress_random as sr  # noqa: E402

from sqp_craft import ProblemSpec, SqpOptions, solve  # noqa: E402

VARIANTS: dict[str, dict] = {
    "default": {},
    "augmented": dict(hessian_update="augmented"),
    "soc": dict(second_order_correction=True),
    "al": dict(merit="augmented-lagrangian"),
    "vl": dict(violation_limit=10.0),
    "elastic": dict(elastic_mode=True),
    # Declare the family's linear rows (no solver option).
    "linear": dict(_linear=True),
    "scale-linear": dict(scaling="linear"),
    "scale-jacobian": dict(scaling="jacobian"),
}

COUNTERS = ["bfgs_damped_updates", "bfgs_modified_updates",
            "hessian_resets", "restoration_steps", "elastic_steps", "second_order_steps",
            "undefined_trials", "violation_limit_rejections", "linear_projection"]


def generated(problem, seeds, count=300, linear=lambda spec: []):
    for seed in seeds:
        rng = sr.np.random.default_rng(seed)
        for k in range(count):
            spec, model, start = problem(rng)
            yield (f"{seed}/{k}", cb.Case("generated", spec, model, start, 0.0, math.inf),
                   linear(spec))


def declared(spec: ProblemSpec, rows: list[int]) -> ProblemSpec:
    """`spec` with `rows` declared linear."""
    _, kwargs = spec.__getnewargs_ex__()
    return ProblemSpec(**{**kwargs, "linear_rows": rows})


def maratos():
    for k, (spec, model, start) in enumerate(cc.maratos_problems()):
        yield str(k), cb.Case("maratos", spec, model, start, 0.0, math.inf), []


def baseline():
    for k, case in enumerate(cb.cases()):
        if case.name == "equality":
            rows = [0]  # x₀ + x₁ = 1
        elif case.name.startswith("orbit"):
            rows = [case.spec.m - 2]  # u(t_f) = 0
        else:
            rows = []
        yield f"{case.name}/{k}", case, rows


def rescaled(case: cb.Case, rng, spread: float) -> cb.Case:
    """`case` in badly scaled units: x = S x', c' = T c with S and T random
    powers of ten in [10^-spread, 10^spread]."""
    spec = case.spec
    _, kwargs = spec.__getnewargs_ex__()
    np_ = sr.np
    S = 10.0 ** rng.uniform(-spread, spread, spec.n)
    T = 10.0 ** rng.uniform(-spread, spread, spec.m)
    columns = np_.asarray(kwargs["jacobian_indices"], dtype=int)
    rows = np_.repeat(np_.arange(spec.m), np_.diff(kwargs["jacobian_indptr"]))
    factor = T[rows] * S[columns]

    def scale(bounds, by):
        return [b * f if math.isfinite(b) else b for b, f in zip(bounds, by)]

    new = ProblemSpec(**{**kwargs,
                         "lower": scale(kwargs["lower"], 1 / S), "upper": scale(kwargs["upper"], 1 / S),
                         "constraint_lower": scale(kwargs["constraint_lower"], T),
                         "constraint_upper": scale(kwargs["constraint_upper"], T)})

    class Model:
        def __init__(self):
            self.inner = case.make_model()

        def functions(self, x):
            f, c = self.inner.functions(list(S * np_.asarray(x)))
            return f, list(T * np_.asarray(c, dtype=float))

        def derivatives(self, x):
            g, j = self.inner.derivatives(list(S * np_.asarray(x)))
            return list(S * np_.asarray(g, dtype=float)), list(factor * np_.asarray(j, dtype=float))

    start = [min(max(v / s, lo), up) for v, s, lo, up in
             zip(case.start, S, new.lower, new.upper)]
    return cb.Case(case.name + "-scaled", new, Model, start, case.reference, case.reference_tolerance,
                   case.max_iterations)


def scaled_family(cases, seed: int, spread: float, copies: int = 1):
    rng = sr.np.random.default_rng(seed)
    for key, case, rows in cases:
        for copy in range(copies):
            yield f"{key}/{copy}", rescaled(case, rng, spread), rows


def orbit_cases():
    for k, case in enumerate(cb.cases()):
        if case.name.startswith("orbit"):
            yield f"{case.name}/{k}", case, [case.spec.m - 2]


FAMILIES = {
    "stress": lambda: generated(sr.problem, range(1, 6)),
    "mpcc": lambda: generated(sm.problem, range(1, 3), linear=lambda spec: [spec.m - 1]),
    "maratos": maratos,
    "baseline": baseline,
    "linear": lambda: generated(sl.problem, range(1, 3),
                                linear=lambda spec: list(range(spec.m - 2))),
    # Badly scaled copies: random powers of ten in [10^-2, 10^2].
    "stress-scaled": lambda: scaled_family(generated(sr.problem, range(1, 3)), 11, 2.0),
    "linear-scaled": lambda: scaled_family(
        generated(sl.problem, range(1, 3), linear=lambda spec: list(range(spec.m - 2))), 12, 2.0),
    "orbit-scaled": lambda: scaled_family(orbit_cases(), 13, 2.0, copies=20),
}


def options_for(variant: str, qp_solver: str) -> tuple[dict, bool]:
    """The SQP settings of `variant` and whether it declares linear rows."""
    settings = dict(max_iterations=300, qp_solver=qp_solver, record_history=True)
    for part in variant.split("+"):
        if part not in VARIANTS:
            sys.exit(f"unknown variant {part!r}; known: {', '.join(VARIANTS)}")
        settings.update(VARIANTS[part])
    return settings, bool(settings.pop("_linear", False))


def run_family(name: str, variants: list[str], qp_solver: str, false_claims: list) -> dict:
    cases = list(FAMILIES[name]())
    results = {}
    for variant in variants:
        settings, linear = options_for(variant, qp_solver)
        options = SqpOptions(**settings)
        records = []
        for key, case, rows in cases:
            spec = declared(case.spec, rows) if linear and rows else case.spec
            r = solve(spec, case.make_model(), case.start, method="sqp",
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
                scales=(min(r.report.variable_scales, default=1.0),
                        max(r.report.variable_scales, default=1.0)),
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
