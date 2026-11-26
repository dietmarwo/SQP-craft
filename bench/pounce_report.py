"""Summarize the pinned POUNCE curvature panel from the worker's evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

from .report import BEST_TOLERANCE, table
from . import pounce_adapter, s2mpj
from .adapter import S2mpjProblem
from .run import configuration_digest
from .worker import objective_check, verify

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "docs/benchmarks/pounce-curvature/selection.json"
WHEEL_PIN = ROOT / "docs/benchmarks/pounce-curvature/pounce-linux-x86_64.txt"
PRIMARY = ("auto", "pounce-sqp-exact", "pounce-sqp-lbfgs")
CONTROLS = ("pounce-sqp-damped", "clarabel", "clarabel+exact")


def pinned_wheel_sha256():
    return WHEEL_PIN.read_text().split("--hash=sha256:")[1].split()[0]


def read_text(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return stream.read()
    return path.read_text()


def completed_fixed(row):
    """Fixed-variable multipliers the checker supplied for a credited solver certificate."""
    if not row["solved"] or row["check"]["certificate"] != "solver":
        return 0
    return (row["check"].get("solver_certificate") or {}).get("completed_fixed_multipliers", 0)


def read_panel(path, names, arms):
    entries = [json.loads(line) for line in read_text(path).splitlines() if line.strip()]
    environments = [row for row in entries if row["event"] == "environment"]
    records = [row for row in entries if row["event"] == "result"]
    found = Counter((row["problem"], row["solver"]) for row in records)
    expected = set(itertools.product(names, arms))
    duplicates = sorted(pair for pair, count in found.items() if count > 1)
    if expected != found.keys() or duplicates:
        raise ValueError(f"{path}: missing {sorted(expected - found.keys())}, "
                         f"extra {sorted(found.keys() - expected)}, duplicates {duplicates}")
    hashes = {info.get("configuration_sha256") for info in environments}
    if len(hashes) != 1 or None in hashes:
        raise ValueError(f"{path}: expected one recorded run configuration")
    for info in environments:
        if info["configuration_sha256"] != configuration_digest(info):
            raise ValueError(f"{path}: inconsistent configuration digest")
    info = environments[0]
    if (info["s2mpj_commit"] != s2mpj.COMMIT or info.get("pounce") != pounce_adapter.VERSION
            or info.get("pounce_reference_commit") != pounce_adapter.COMMIT):
        raise ValueError(f"{path}: wrong S2MPJ or POUNCE revision")
    if info["selection_sha256"] != hashlib.sha256(SELECTION.read_bytes()).hexdigest():
        raise ValueError(f"{path}: wrong frozen selection")
    if info["cases"] != list(names) or info["solver_order"] != list(arms):
        raise ValueError(f"{path}: recorded cases or solvers differ from the panel")
    for row in records:
        if row.get("x") is not None and not row.get("start_sha256"):
            raise ValueError(f"{path}: candidate without a start hash")
    for name in names:
        starts = {row["start_sha256"] for row in records
                  if row["problem"] == name and "start_sha256" in row}
        if len(starts) > 1:
            raise ValueError(f"{path}: different inputs for {name}")
    return table(records), environments[0]


def pair_summary(name_a, name_b, rows):
    only_a, only_b, neither, different, zero = [], [], [], [], []
    shared = 0
    ratios, callbacks_a, callbacks_b = [], [], []
    hessians_a, hessians_b = [], []
    actual_ratios, solver_dual_ratios, complete_dual_ratios = [], [], []
    wins = Counter()
    for problem, (a, b) in rows.items():
        if not a["solved"] or not b["solved"]:
            if a["solved"]:
                only_a.append(f"{problem} ({b['status']})")
            elif b["solved"]:
                only_b.append(f"{problem} ({a['status']})")
            else:
                neither.append(f"{problem} ({a['status']} / {b['status']})")
            continue
        shared += 1
        fa, fb = a["check"]["objective"], b["check"]["objective"]
        if abs(fa - fb) > BEST_TOLERANCE * max(1.0, abs(min(fa, fb))):
            different.append(f"{problem}: {fa:.8g} / {fb:.8g}")
            continue
        if not (a["callbacks"] and b["callbacks"]):
            zero.append(problem)
            continue
        ratios.append(a["callbacks"] / b["callbacks"])
        ea, eb = (row["callback_work"]["evaluations"] for row in (a, b))
        attempts_ratio = (ea["functions"] + ea["derivatives"]) / (eb["functions"] + eb["derivatives"])
        actual_ratios.append(attempts_ratio)
        if all(row["check"]["certificate"] == "solver" for row in (a, b)):
            solver_dual_ratios.append(attempts_ratio)
            if not any(completed_fixed(row) for row in (a, b)):
                complete_dual_ratios.append(attempts_ratio)
        wins["A" if a["callbacks"] < b["callbacks"]
             else "B" if a["callbacks"] > b["callbacks"] else "equal"] += 1
        callbacks_a.append(sum(a["callback_work"]["seconds"].values()))
        callbacks_b.append(sum(b["callback_work"]["seconds"].values()))
        hessians_a.append(a["hessians"])
        hessians_b.append(b["hessians"])
    summary = [
        f"**{name_a} / {name_b}**",
        "",
        f"- Both verified: {shared}. A only: {', '.join(only_a) or 'none'}. "
        f"B only: {', '.join(only_b) or 'none'}. Neither: {', '.join(neither) or 'none'}.",
        f"- Different verified objectives: {', '.join(different) or 'none'}. "
        f"Zero first-order work: {', '.join(zero) or 'none'}.",
    ]
    if ratios:
        summary.append(f"- On {len(ratios)} comparable points, median paired F+D ratio "
                       f"{median(ratios):.3f} (A/B). F+D wins and ties {wins['A']}/{wins['B']}/{wins['equal']} "
                       f"(A/B/tie). Median H counts {median(hessians_a):g}/{median(hessians_b):g}. "
                       f"Median model callback times {1e3 * median(callbacks_a):.3g}/"
                       f"{1e3 * median(callbacks_b):.3g} ms in the panel.")
        summary.append(f"- Counting actual grouped F+D attempts, median paired ratio "
                       f"{median(actual_ratios):.3f} ({len(actual_ratios)} pairs).")
        if solver_dual_ratios:
            summary.append(f"- Restricting to points verified with both solvers’ supplied duals, "
                           f"median paired grouped-attempt ratio {median(solver_dual_ratios):.3f} "
                           f"({len(solver_dual_ratios)} pairs).")
        if complete_dual_ratios and len(complete_dual_ratios) != len(solver_dual_ratios):
            summary.append(f"- Restricting further to pairs where neither solver left out a "
                           f"fixed-variable multiplier, {median(complete_dual_ratios):.3f} "
                           f"({len(complete_dual_ratios)} pairs).")
    return summary + [""]


def metrics(row):
    work = [row.get(key) for key in ("functions", "derivatives", "hessians")]
    counts = "/".join(str(value) if value is not None else "—" for value in work)
    state = ("verified" if row["claimed"] else "verified at " + row["status"]
             ) if row["solved"] else row["status"]
    work = row.get("callback_work")
    if work is not None:
        actual = work["evaluations"]["functions"] + work["evaluations"]["derivatives"]
        if actual != row["functions"] + row["derivatives"]:
            counts += f" ({actual} F+D attempts)"
    if row["false_claim"]:
        state = f"rejected {state}"
    if row["solved"] and row["check"]["certificate"] == "least-squares":
        state += " (LS)"
    elif completed_fixed(row):
        state += " (fixed)"
    return f"{state}, {counts}"


def render(selection, primary, controls, fixtures, info):
    lines = [
        "# POUNCE curvature pilot",
        "",
        f"SQP-craft {info['sqp_craft']} at `{info['sqp_craft_commit'][:10]}`, "
        f"POUNCE {info.get('pounce')}, S2MPJ `{info['s2mpj_commit'][:10]}`. "
        f"{info['cpu']}, Python {info['python']}, NumPy {info['numpy']}, SciPy {info['scipy']}.",
        "",
        "Solved means independently accepted first-order KKT. LS denotes a recovered least-squares "
        "certificate. Supplied duals counts points certified by the solver's own multipliers; "
        "the next column gives how many of them lacked multipliers of fixed variables, which the "
        "checker took from stationarity. F and D are distinct grouped value and derivative points. H counts analytic "
        "Lagrangian-Hessian calls, including failed attempts. The exact callbacks emit a full lower "
        "triangle on this panel. The evidence includes solver duals, residuals and model timings.",
        "",
        "## Outcomes",
        "",
        "| Solver | Runs | Verified | Supplied duals | Of these, fixed variables completed | Recovered duals | "
        "Strict and verified | Verified without strict claim | "
        "False strict claims | Acceptable-level (rejected) | Objective mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for runs, arms in ((primary, PRIMARY), (controls, CONTROLS)):
        for arm in arms:
            rows = [pair[arm] for pair in runs.values()]
            accepted = sum(row["solved"] for row in rows)
            strict = sum(row["solved"] and row["claimed"] for row in rows)
            acceptable = sum(bool(row.get("acceptable")) for row in rows)
            rejected_acceptable = sum(bool(row.get("acceptable")) and not row["solved"] for row in rows)
            mismatch = sum(row.get("objective_check", {}).get("mismatch") is True for row in rows)
            supplied = sum(row["solved"] and row["check"]["certificate"] == "solver" for row in rows)
            completed = sum(completed_fixed(row) > 0 for row in rows)
            lines.append(f"| {arm} | {len(rows)} | {accepted} | {supplied} | {completed} | {accepted - supplied} | "
                         f"{strict} | {accepted - strict} | "
                         f"{sum(row['false_claim'] for row in rows)} | "
                         f"{acceptable} ({rejected_acceptable}) | {mismatch} |")
    lines += ["", "### Terminal solver statuses", ""]
    for runs, arms in ((primary, PRIMARY), (controls, CONTROLS)):
        for arm in arms:
            statuses = Counter(pair[arm]["status"] for pair in runs.values())
            lines.append(f"- {arm} — " + ", ".join(f"{status} {count}" for status, count in statuses.items()) + ".")
    lines += ["", "## Paired primary results", ""]
    for a, b in itertools.combinations(PRIMARY, 2):
        lines += pair_summary(a, b, {p: (rows[a], rows[b]) for p, rows in primary.items()})
    lines += ["## All 20 cases", "",
              "| Case | n, m | SQP-craft BFGS | POUNCE exact | POUNCE L-BFGS |",
              "|---|---:|---|---|---|"]
    for case in selection["problems"]:
        name = case["name"]
        values = " | ".join(metrics(primary[name][arm]) for arm in PRIMARY)
        lines.append(f"| {name} | {case['n']}, {case['m']} | {values} |")
    lines += ["", "## Controls", "",
              "| Case | POUNCE damped BFGS | SQP-craft Clarabel BFGS | SQP-craft Clarabel exact | "
              "Native exact diagonal shifts | Largest shift |",
              "|---|---|---|---|---:|---:|"]
    for name in selection["controls"]:
        values = " | ".join(metrics(controls[name][arm]) for arm in CONTROLS)
        structure = controls[name]["clarabel+exact"].get("report", {}).get("structure") or {}
        shift = structure.get("maximum_diagonal_shift")
        lines.append(f"| {name} | {values} | {structure.get('diagonal_shifts', '—')} | "
                     f"{format(shift, '.6g') if shift is not None else '—'} |")
    lines += ["", "## Release fixtures", ""]
    for fixture in fixtures:
        lines.append(f"- POUNCE #{fixture['id']} — {'pass' if fixture['passed'] else 'fail'}.")
    return "\n".join(lines) + "\n"


def quality_error(row):
    if row["false_claim"]:
        return "strict success failed independent KKT verification"
    if row["status"] in {"timeout", "crash", "error", "evaluation_error", "unsupported"}:
        return row["status"]
    if row.get("x") is not None and row.get("objective_check", {}).get("mismatch") is not False:
        return "reported objective missing or inconsistent with fresh evaluation"
    return None


def recheck_candidate(problem, row):
    """Replay verification without invoking either optimizer."""
    if row.get("x") is None:
        if row["check"]["accepted"]:
            raise ValueError("accepted record without a candidate")
        return
    pair = f"{row['problem']} / {row['solver']}"
    expected_start = hashlib.sha256(problem.x0.astype("<f8").tobytes()).hexdigest()
    if row["start_sha256"] != expected_start:
        raise ValueError(f"{pair}: start hash differs from original model")
    if row["solver"] in pounce_adapter.MODES:
        lam, z, _ = pounce_adapter.duals(row["raw_multipliers"], problem.n, problem.m)
        if (lam, z) != (row["constraint_multipliers"], row["variable_multipliers"]):
            raise ValueError(f"{pair}: stored dual conversion differs from raw multipliers")
    fresh = verify(problem, row)
    saved = row["check"]
    for key in ("accepted", "reason", "certificate"):
        if fresh.get(key) != saved.get(key):
            raise ValueError(f"{pair}: replay differs for {key}")
    for current, old in ((fresh, saved),
                         (fresh.get("solver_certificate"), saved.get("solver_certificate"))):
        if current is None or old is None:
            if current != old:
                raise ValueError(f"{pair}: missing certificate on replay")
            continue
        for key in ("objective", "primal", "gradient_scale", "stationarity", "complementarity",
                    "dual_sign", "duals_passed", "completed_fixed_multipliers"):
            if key in current or key in old:
                if (key not in current or key not in old or
                        not math.isclose(current[key], old[key], rel_tol=1e-8, abs_tol=1e-12)):
                    raise ValueError(f"{pair}: replay differs for {key}")
    match = objective_check(row.get("reported_objective"), fresh.get("objective"))
    old_match = row["objective_check"]
    delta, old_delta = match["delta"], old_match["delta"]
    same_delta = (delta == old_delta if delta is None or old_delta is None else
                  math.isclose(delta, old_delta, rel_tol=1e-8, abs_tol=1e-12))
    if not same_delta or match["mismatch"] != old_match["mismatch"]:
        raise ValueError(f"{pair}: objective comparison differs on replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("controls", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="exit nonzero for false claims, objective failures or incomplete runs")
    parser.add_argument("--recheck", action="store_true",
                        help="re-evaluate saved candidates on the original S2MPJ models")
    args = parser.parse_args()

    selection = json.loads(SELECTION.read_text())
    names = [case["name"] for case in selection["problems"]]
    primary, info = read_panel(args.primary, names, PRIMARY)
    controls, control_info = read_panel(args.controls, selection["controls"], CONTROLS)
    shared_keys = ("python", "numpy", "scipy", "platform", "cpu", "jobs", "source_sha256", "selection_sha256", "sqp_craft_commit", "sqp_craft_binary_sha256", "sqp_craft_python_sha256",
                   "pounce", "pounce_wheel_sha256", "sqp_craft_callback_limit", "max_iterations",
                   "solve_seconds_limit", "watchdog_seconds", "threads")
    if any(info.get(key) != control_info.get(key) for key in shared_keys):
        raise ValueError("primary and controls use different inputs or software")
    for name in selection["controls"]:
        starts = {row["start_sha256"] for row in (*primary[name].values(), *controls[name].values())
                  if "start_sha256" in row}
        if len(starts) > 1:
            raise ValueError(f"control start differs on {name}")
    fixtures = json.loads(args.fixtures.read_text())
    if sorted(fixture["id"] for fixture in fixtures) != ["922", "939"]:
        raise ValueError("expected both 0.12.0 release fixtures")
    print(render(selection, primary, controls, fixtures, info), end="")
    if args.recheck:
        for case in selection["problems"]:
            name = case["name"]
            model = S2mpjProblem(s2mpj.load(name))
            if (model.n, model.m) != (case["n"], case["m"]):
                raise ValueError(f"{name}: replay dimensions disagree with selection")
            for row in (*primary[name].values(), *controls.get(name, {}).values()):
                recheck_candidate(model, row)
    if args.check:
        errors = [f"fixture #{f['id']} failed" for f in fixtures if f["passed"] is not True]
        for label, header in (("primary", info), ("controls", control_info)):
            if header.get("pounce_wheel_sha256") != pinned_wheel_sha256():
                errors.append(f"{label}: the recorded POUNCE wheel is missing or not the pinned one")
        for panel in (primary, controls):
            for pairs in panel.values():
                for row in pairs.values():
                    reason = quality_error(row)
                    if reason:
                        errors.append(f"{row['problem']} / {row['solver']}: {reason}")
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
