"""Tables from the raw results: counts, costs, performance profiles, pairwise
comparisons and failure classes.

    python -m bench.report selection.json results.jsonl[.gz] tables.md

A run is *solved* when the independent check passes (bench/check.py), and
*solved to the best objective* when, in addition, its objective is within
10⁻⁶·max(1, |f*|) of the best objective f* of any solved run of the problem.
A *false claim* is a claimed success that fails the check. Costs are
callbacks (function plus derivative evaluations, as distinct points) and
wall time; a failure has infinite cost.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import pathlib

TAUS = [1, 2, 4, 8, 16, 32, math.inf]
DEFAULT = "clarabel"
BEST_TOLERANCE = 1e-6

# The decisions: (question, current choice, alternative).
DECISIONS = [
    ("Q1 backend", "clarabel", "active-set"),
    ("Q2 merit: second-order correction", "active-set", "active-set+soc"),
    ("Q2 merit: augmented Lagrangian", "active-set", "active-set+al"),
    ("Q2 merit: augmented Lagrangian (Clarabel)", "clarabel", "clarabel+al"),
    ("Q3 BFGS repair", "active-set", "active-set+augmented"),
    ("Q4 scaling from the Jacobian", "active-set", "active-set+scale-jacobian"),
    ("Q4 scaling from the linear rows", "active-set+linear", "active-set+linear+scale-linear"),
    ("Q5 linear rows declared", "active-set", "active-set+linear"),
    ("Q6 elastic mode", "active-set", "active-set+elastic"),
    ("SNOPT-like combination", "active-set", "active-set+snopt"),
]
SQP_CRAFT = ["clarabel", "active-set", "active-set+soc", "active-set+al",
             "active-set+augmented", "active-set+scale-jacobian", "active-set+linear",
             "active-set+linear+scale-linear", "active-set+elastic", "active-set+snopt",
             "clarabel+al"]
COMPARATORS = ["slsqp", "uno-filtersqp", "uno-ipopt", "uno-filtersqp-exact"]

# Published optimal values (Hock & Schittkowski 1981) for spot checks.
KNOWN_OPTIMA = {
    "HS6": 0.0, "HS7": -1.7320508, "HS10": -1.0, "HS14": 1.3934650, "HS21": -99.96,
    "HS26": 0.0, "HS27": 0.04, "HS35": 0.1111111, "HS39": -1.0, "HS40": -0.25,
    "HS43": -44.0, "HS44": -15.0, "HS46": 0.0, "HS71": 17.0140173, "HS76": -4.6818182,
    "HS77": 0.24150513, "HS78": -2.9197004, "HS79": 0.0787768, "HS100": 680.6300573,
    "HS113": 24.3062091,
}


# --- loading ------------------------------------------------------------------
def read(path: pathlib.Path) -> tuple[list[dict], list[dict]]:
    """(results, environment lines); the last result of each pair wins."""
    opener = gzip.open if path.suffix == ".gz" else open
    results: dict = {}
    environments = []
    with opener(path, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "environment":
                environments.append(record)
            elif record.get("event") == "result":
                results[(record["problem"], record["solver"])] = record
    return list(results.values()), environments


def table(records: list[dict]) -> dict:
    """{problem: {solver: record}} with `solved`, `best`, `false_claim`,
    `callbacks` added to every record."""
    by_problem: dict = collections.defaultdict(dict)
    for r in records:
        check = r.get("check") or {}
        r["solved"] = bool(check.get("accepted"))
        r["false_claim"] = bool(r.get("claimed")) and not r["solved"]
        r["callbacks"] = ((r.get("functions") or 0) + (r.get("derivatives") or 0)
                          if r["solved"] else math.inf)
        r["time"] = r.get("seconds", math.inf) if r["solved"] else math.inf
        by_problem[r["problem"]][r["solver"]] = r
    for runs in by_problem.values():
        solved = [r["check"]["objective"] for r in runs.values() if r["solved"]]
        best = min(solved) if solved else None
        for r in runs.values():
            r["best"] = (r["solved"] and best is not None and
                         r["check"]["objective"] <= best + BEST_TOLERANCE * max(1.0, abs(best)))
    return dict(by_problem)


# --- measures -----------------------------------------------------------------
def shifted_geometric_mean(values, shift: float) -> float:
    values = list(values)
    if not values:
        return math.nan
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def profile(costs: dict[str, list[float]], taus=TAUS) -> dict[str, list[float]]:
    """Dolan–Moré: for each solver, the fraction of problems whose cost is
    within τ of the best solver's on that problem. `costs[solver][k]` is the
    cost on problem k (infinite for a failure)."""
    solvers = list(costs)
    count = len(next(iter(costs.values()))) if costs else 0
    best = [min(costs[s][k] for s in solvers) for k in range(count)]
    out = {}
    for s in solvers:
        ratios = [costs[s][k] / best[k] if math.isfinite(costs[s][k]) else math.inf
                  for k in range(count)]
        out[s] = [sum(math.isfinite(r) and r <= t for r in ratios) / count if count
                  else math.nan for t in taus]
    return out


def pairwise(by_problem: dict, a: str, b: str) -> dict:
    """Problems solved by only one of `a` and `b`, and the costs on those both
    solve."""
    only_a, only_b, both = [], [], []
    for name, runs in sorted(by_problem.items()):
        ra, rb = runs.get(a), runs.get(b)
        if ra is None or rb is None:
            continue
        if ra["solved"] and rb["solved"]:
            both.append((ra, rb))
        elif ra["solved"]:
            only_a.append(name)
        elif rb["solved"]:
            only_b.append(name)
    wins = sum(rb["callbacks"] < ra["callbacks"] for ra, rb in both)
    losses = sum(rb["callbacks"] > ra["callbacks"] for ra, rb in both)
    best_a = sum(1 for runs in by_problem.values() if a in runs and runs[a]["best"])
    best_b = sum(1 for runs in by_problem.values() if b in runs and runs[b]["best"])
    return dict(
        only_a=only_a, only_b=only_b, both=len(both), wins=wins, losses=losses,
        ties=len(both) - wins - losses, best_a=best_a, best_b=best_b,
        callbacks_a=shifted_geometric_mean([ra["callbacks"] for ra, _ in both], 10.0),
        callbacks_b=shifted_geometric_mean([rb["callbacks"] for _, rb in both], 10.0),
        time_a=shifted_geometric_mean([ra["time"] for ra, _ in both], 1.0),
        time_b=shifted_geometric_mean([rb["time"] for _, rb in both], 1.0),
        iterations_a=shifted_geometric_mean([ra["iterations"] or 0 for ra, _ in both], 10.0),
        iterations_b=shifted_geometric_mean([rb["iterations"] or 0 for _, rb in both], 10.0))


def tolerance_sensitivity(record: dict, factor: float) -> bool:
    """Whether `record` passes the check with every threshold times `factor`
    (from the stored residuals; the multipliers are not re-fitted)."""
    c = record.get("check") or {}
    if "primal" not in c:
        return False
    limit = 1e-6 * factor
    return (c["primal"] <= limit and
            max(c["stationarity"], c["complementarity"], c["dual_sign"])
            <= limit * c["gradient_scale"])


# --- formatting -----------------------------------------------------------------
def _fmt(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        if math.isinf(v):
            return "∞"
        return f"{v:.{digits}g}" if abs(v) < 1e4 else f"{v:.0f}"
    return str(v)


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return lines + [""]


def _short(names: list[str], limit: int = 12) -> str:
    if not names:
        return "–"
    text = ", ".join(names[:limit])
    return text + (f" (+{len(names) - limit})" if len(names) > limit else "")


def render(selection: dict, records: list[dict], environments: list[dict]) -> str:
    by_problem = table(records)
    meta = {e["name"]: e for e in selection["problems"]}
    names = sorted(n for n in by_problem if n in meta)
    by_problem = {n: by_problem[n] for n in names}
    solvers = [s for s in SQP_CRAFT + COMPARATORS
               if any(s in runs for runs in by_problem.values())]
    subsets = ["unconstrained", "bounds", "linear", "nonlinear"]
    out = ["# Benchmark tables", "",
           "Generated by `python -m bench.report`; the analysis is in "
           "[benchmark-results.md](../benchmark-results.md).", ""]
    if environments:
        e = environments[0]
        out += [f"- sqp-craft {e.get('sqp_craft')} at `{(e.get('sqp_craft_commit') or '?')[:10]}`, "
                f"S2MPJ `{(e.get('s2mpj_commit') or '?')[:10]}`, Uno {e.get('uno')}, "
                f"SciPy {e.get('scipy')}, NumPy {e.get('numpy')}, Python {e.get('python')}",
                f"- {e.get('cpu')}, {e.get('cpus')} CPUs, two solves in parallel", ""]
    counts = collections.Counter(meta[n]["subset"] for n in names)
    control = [n for n in names if meta[n]["control"]]
    out += [f"{len(names)} problems: " + ", ".join(f"{counts[s]} {s}" for s in subsets)
            + f"; {len(control)} control-like.", ""]

    # Counts.
    out += ["## Solved, solved to the best objective, false claims", ""]
    rows = []
    for s in solvers:
        runs = [by_problem[n][s] for n in names if s in by_problem[n]]
        row = [s, len(runs), sum(r["solved"] for r in runs), sum(r["best"] for r in runs)]
        for sub in subsets:
            row.append(sum(r["solved"] for r in runs if meta[r["problem"]]["subset"] == sub))
        row.append(sum(r["solved"] for r in runs if meta[r["problem"]]["control"]))
        row += [sum(r["false_claim"] for r in runs),
                sum(r["solved"] and not r.get("claimed") for r in runs)]
        rows.append(row)
    out += _table(["solver", "runs", "solved", "best"] + subsets
                  + ["control", "false claims", "solved, not claimed"], rows)
    out += ["Subset sizes: " + ", ".join(f"{s} {counts[s]}" for s in subsets)
            + f", control {len(control)}.", ""]

    # Virtual best of the sqp-craft variants.
    variants = [s for s in SQP_CRAFT if s in solvers]
    any_variant = [n for n in names if any(by_problem[n].get(s, {}).get("solved") for s in variants)]
    any_comparator = [n for n in names if any(by_problem[n].get(s, {}).get("solved")
                                              for s in COMPARATORS)]
    out += [f"Some sqp-craft variant solves {len(any_variant)} problems; some comparator "
            f"solves {len(any_comparator)}; together {len(set(any_variant) | set(any_comparator))}; "
            f"no solver solves {len(names) - len(set(any_variant) | set(any_comparator))}.", ""]

    # Tolerance sensitivity.
    out += ["## Sensitivity to the check's thresholds", "",
            "Solved counts with every threshold of the check multiplied by a factor, from the "
            "stored residuals (multipliers not re-fitted, so the loose columns are lower bounds).",
            ""]
    factors = [0.1, 1.0, 10.0, 100.0]
    rows = [[s] + [sum(tolerance_sensitivity(by_problem[n][s], f)
                       for n in names if s in by_problem[n]) for f in factors] for s in solvers]
    out += _table(["solver"] + [f"× {f:g}" for f in factors], rows)

    # Costs.
    out += ["## Costs on the problems every listed solver solves", ""]
    for group, members in (("sqp-craft variants", variants),
                           ("Comparators and the two backends",
                            [s for s in ["clarabel", "active-set"] + COMPARATORS if s in solvers])):
        common = [n for n in names if all(by_problem[n].get(s, {}).get("solved") for s in members)]
        rows = []
        for s in members:
            runs = [by_problem[n][s] for n in common]
            rows.append([s, shifted_geometric_mean([r["iterations"] or 0 for r in runs], 10.0),
                         shifted_geometric_mean([r["functions"] for r in runs], 10.0),
                         shifted_geometric_mean([r["derivatives"] for r in runs], 10.0),
                         shifted_geometric_mean([r["callbacks"] for r in runs], 10.0),
                         shifted_geometric_mean([r["time"] for r in runs], 1.0)])
        out += [f"**{group}**, {len(common)} problems; shifted geometric means (shift 10, "
                "time 1 s):", ""]
        out += _table(["solver", "iterations", "functions", "derivatives", "callbacks",
                       "time (s)"], rows)

    # Profiles.
    out += ["## Performance profiles", "",
            "Fraction of all problems solved within τ times the best listed solver's cost "
            "(Dolan & Moré 2002). With more than two solvers they rank only the best "
            "(Gould & Scott 2016); the pairwise tables decide.", ""]
    for group, members in (("sqp-craft variants", variants),
                           ("Comparators", [s for s in ["clarabel", "active-set",
                                                        "active-set+snopt"] + COMPARATORS
                                            if s in solvers])):
        for measure in ("callbacks", "time"):
            costs = {s: [by_problem[n].get(s, {}).get(measure, math.inf) for n in names]
                     for s in members}
            prof = profile(costs)
            out += [f"**{group}, {measure}:**", ""]
            out += _table(["solver"] + [f"τ={_fmt(float(t))}" for t in TAUS],
                          [[s] + prof[s] for s in members])

    # Pairwise.
    out += ["## Pairwise comparisons", "",
            "For each decision, the current choice A against the alternative B. *Wins* are "
            "problems both solve on which B needs fewer callbacks; the means are shifted "
            "geometric means over the problems both solve.", ""]
    rows = []
    for question, a, b in DECISIONS:
        if a not in solvers or b not in solvers:
            continue
        p = pairwise(by_problem, a, b)
        rows.append([question, f"{a} → {b}", len(p["only_a"]), len(p["only_b"]), p["both"],
                     f"{p['wins']}/{p['losses']}/{p['ties']}", p["best_a"], p["best_b"],
                     f"{_fmt(p['callbacks_a'])} → {_fmt(p['callbacks_b'])}",
                     f"{_fmt(p['iterations_a'])} → {_fmt(p['iterations_b'])}",
                     f"{_fmt(p['time_a'])} → {_fmt(p['time_b'])}"])
    out += _table(["decision", "A → B", "only A", "only B", "both", "B wins/losses/ties",
                   "best A", "best B", "callbacks", "iterations", "time (s)"], rows)
    out += ["Problems solved by only one side:", ""]
    for question, a, b in DECISIONS:
        if a in solvers and b in solvers:
            p = pairwise(by_problem, a, b)
            out.append(f"- {question}: only {a}: {_short(p['only_a'])}; only {b}: "
                       f"{_short(p['only_b'])}")
    out.append("")
    for a, b in (("clarabel", "uno-filtersqp"), ("active-set", "uno-filtersqp"),
                 ("active-set", "slsqp"), ("active-set", "uno-ipopt"),
                 ("uno-filtersqp", "uno-filtersqp-exact")):
        if a in solvers and b in solvers:
            p = pairwise(by_problem, a, b)
            out.append(f"- {a} vs {b}: only {a} {len(p['only_a'])}, only {b} {len(p['only_b'])}, "
                       f"both {p['both']}; callbacks {_fmt(p['callbacks_a'])} vs "
                       f"{_fmt(p['callbacks_b'])}; time {_fmt(p['time_a'])} vs "
                       f"{_fmt(p['time_b'])} s; best {p['best_a']} vs {p['best_b']}")
    out.append("")

    # Spot checks.
    known = [n for n in names if n in KNOWN_OPTIMA]
    if known:
        out += ["## Spot checks against published optima", "",
                "Best objective of any solved run against Hock & Schittkowski (1981), and how "
                "many runs reach the published value within 10⁻⁶ relative (the values are "
                "printed to 7–8 digits).", ""]
        rows = []
        for n in known:
            solved = [r for r in by_problem[n].values() if r["solved"]]
            best = min((r["check"]["objective"] for r in solved), default=None)
            target = KNOWN_OPTIMA[n]
            near = sum(abs(r["check"]["objective"] - target) <= 1e-6 * max(1.0, abs(target))
                       + 5e-8 for r in solved)
            rows.append([n, f"{target:.9g}", "–" if best is None else f"{best:.9g}",
                         f"{near}/{len(by_problem[n])}"])
        out += _table(["problem", "published", "best found", "runs at the published value"],
                      rows)

    # Failure classes.
    infeasible = {n for n in names
                  if any("INFEASIBLE_STATIONARY" in str(r.get("status")) or
                         r.get("status") == "locally_infeasible"
                         for r in by_problem[n].values())}
    unsolved = [n for n in names if not any(r["solved"] for r in by_problem[n].values())]
    out += ["## Failure classes", "",
            f"{len(unsolved)} problems no run solves; on {len(set(unsolved) & infeasible)} of "
            "them some solver reports a stationary point of the infeasibility (Uno's "
            "`INFEASIBLE_STATIONARY_POINT` or sqp-craft's `locally_infeasible`), so they are "
            "probably infeasible from their start: "
            f"{_short(sorted(set(unsolved) & infeasible), 30)}.", "",
            "Status of every unsolved run. *Comparator* and *variant* count the problems some "
            "other comparator or some sqp-craft variant solves; *infeasible?* those on which some "
            "solver reports infeasibility.", ""]
    for s in solvers:
        failed = [by_problem[n][s] for n in names if s in by_problem[n]
                  and not by_problem[n][s]["solved"]]
        if not failed:
            continue
        statuses = collections.Counter(r["status"] for r in failed)

        def count(status, among):
            return sum(1 for r in failed if r["status"] == status and r["problem"] in among)

        out.append(f"**{s}** ({len(failed)} unsolved):")
        out.append("")
        out += _table(["status", "runs", "comparator", "variant", "infeasible?", "examples"],
                      [[st, k, count(st, any_comparator), count(st, any_variant),
                        count(st, infeasible),
                        _short([r["problem"] for r in failed if r["status"] == st], 8)]
                       for st, k in statuses.most_common()])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("selection", type=pathlib.Path)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    records, environments = read(args.results)
    args.output.write_text(render(json.loads(args.selection.read_text()), records, environments))
    false_claims = [r for r in records if r["solver"] in SQP_CRAFT and r.get("claimed")
                    and not (r.get("check") or {}).get("accepted")]
    for r in false_claims:
        print(f"false claim: {r['problem']} {r['solver']} {r.get('check')}")
    raise SystemExit(1 if false_claims else 0)


if __name__ == "__main__":
    main()
