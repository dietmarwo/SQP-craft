"""The selection rules of the benchmark, applied to the catalog, and the
subsets the report splits it into.

    python -m bench.selection catalog.json selection.json

Rules (docs/benchmark-harness-plan.md):
- n ≤ 256 (the dense cap) and m ≤ 512;
- smooth class (`R` in the class string);
- defined at the start (f, c, ∇f and J finite there);
- setup within 30 s, and one evaluation of f, c, ∇f and J within 0.2 s.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

MAX_N, MAX_M = 256, 512
MAX_SETUP, MAX_EVALUATION = 30.0, 0.2

# Discretized optimal-control and trajectory problems, by name.
CONTROL_NAMES = re.compile(
    r"^(DTOC\w*|CATMIX|LAUNCH|ROBOT\w*|HANGING|CATENA\w*|AIRCRFT\w*|GLIDER\w*|ROCKET\w*|"
    r"CHAIN\w*|STEERING|TRAIN[FH]|OPTCTRL\w*|OPTCNTRL|HAGER\d+|READING\d+|DRUGDIS\w*|"
    r"CHEMRCT\w*|ORBIT\w*|ELEC|MANNE)$")


def reason(entry: dict) -> str | None:
    """Why `entry` is excluded, or `None`."""
    if not entry.get("defined", False):
        return "undefined at the start: " + str(entry.get("error"))
    if entry["n"] > MAX_N:
        return "n > 256"
    if entry["m"] > MAX_M:
        return "m > 512"
    if not entry.get("smooth", False):
        return "not smooth"
    if entry.get("setup_seconds", 0.0) > MAX_SETUP:
        return "setup over 30 s"
    if entry.get("evaluation_seconds", 0.0) > MAX_EVALUATION:
        return "evaluation over 0.2 s"
    return None


def subset(entry: dict) -> str:
    """unconstrained, bounds, linear or nonlinear."""
    if entry["m"] == 0:
        return "unconstrained" if entry.get("constraint_type") == "U" else "bounds"
    if entry.get("linear_rows", 0) == entry["m"]:
        return "linear"
    return "nonlinear"


def control(name: str) -> bool:
    return bool(CONTROL_NAMES.match(name))


def select(catalog: dict) -> dict:
    selected, excluded = [], {}
    for entry in catalog["problems"]:
        why = reason(entry)
        if why is None:
            selected.append(dict(name=entry["name"], n=entry["n"], m=entry["m"],
                                 subset=subset(entry), control=control(entry["name"]),
                                 linear_rows=entry.get("linear_rows", 0),
                                 pbclass=entry.get("pbclass")))
        else:
            excluded[entry["name"]] = why
    return dict(s2mpj_commit=catalog.get("s2mpj_commit"),
                rules=dict(max_n=MAX_N, max_m=MAX_M, smooth=True, defined=True,
                           max_setup_seconds=MAX_SETUP, max_evaluation_seconds=MAX_EVALUATION),
                problems=sorted(selected, key=lambda e: e["name"]), excluded=excluded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("catalog", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    selection = select(json.loads(args.catalog.read_text()))
    args.output.write_text(json.dumps(selection, indent=1))
    counts: dict = {}
    for e in selection["problems"]:
        counts[e["subset"]] = counts.get(e["subset"], 0) + 1
    print(f"{len(selection['problems'])} selected ({counts}), "
          f"{sum(e['control'] for e in selection['problems'])} control-like, "
          f"{len(selection['excluded'])} excluded")


if __name__ == "__main__":
    main()
