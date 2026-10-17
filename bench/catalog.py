"""Catalog of the S2MPJ problems: sizes, classes and whether each can be set up
and evaluated at its start point.

    python -m bench.catalog out.json [--cache DIR] [--timeout SECONDS] [--jobs N]

Each problem is set up in its own subprocess (default parameters, a time
limit per problem), so a slow or failing setup cannot stop the catalog. The
selection rules of the benchmark are applied to this catalog
(bench/selection.py), which keeps them reproducible.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import sys
import time

from . import s2mpj


def describe(name: str, cache=None) -> dict:
    """Facts about one problem; runs inside a worker subprocess."""
    import numpy as np

    from .adapter import S2mpjProblem

    began = time.perf_counter()
    raw = s2mpj.load(name, cache)
    setup = time.perf_counter() - began
    klass = str(raw.pbclass)
    code = klass.split("-")[1] if klass.count("-") >= 3 else ""
    entry = dict(name=name, pbclass=klass, n=int(raw.n), m=int(getattr(raw, "m", 0) or 0),
                 objective_type=code[1:2], constraint_type=code[2:3], smooth=code[3:4] == "R",
                 setup_seconds=setup)
    problem = S2mpjProblem(raw)
    entry.update(nnz=problem.nnz, linear_rows=len(problem.linear_rows),
                 pattern_extended=problem.pattern_extended,
                 start_clipped=problem.start_clipped)
    began = time.perf_counter()
    try:
        f, c = problem.functions(problem.x0)
        g, j = problem.derivatives(problem.x0)
        finite = bool(np.isfinite(f) and np.all(np.isfinite(c)) and np.all(np.isfinite(g))
                      and np.all(np.isfinite(j)))
        entry.update(defined=finite, error=None if finite else "nonfinite at the start")
    except Exception as error:  # recorded: the problem is excluded
        entry.update(defined=False, error=f"{type(error).__name__}: {error}"[:200])
    entry["evaluation_seconds"] = time.perf_counter() - began
    return entry


def describe_in_subprocess(name: str, cache, timeout: float) -> dict:
    command = [sys.executable, "-c",
               "import json, sys; from bench.catalog import describe; "
               f"print(json.dumps(describe({name!r}, {cache!r})))"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              cwd=pathlib.Path(__file__).resolve().parents[1])
    except subprocess.TimeoutExpired:
        return dict(name=name, defined=False, error=f"setup or evaluation over {timeout:g} s")
    if done.returncode != 0:
        tail = (done.stderr.strip().splitlines() or ["?"])[-1]
        return dict(name=name, defined=False, error=tail[:200])
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--cache")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--only", help="comma-separated problem names")
    args = parser.parse_args()
    names = args.only.split(",") if args.only else s2mpj.problem_names(args.cache)
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        entries = list(pool.map(lambda n: describe_in_subprocess(n, args.cache, args.timeout),
                                names))
    args.output.write_text(json.dumps(dict(s2mpj_commit=s2mpj.COMMIT, problems=entries),
                                      indent=1))
    defined = sum(e.get("defined", False) for e in entries)
    print(f"{len(entries)} problems, {defined} defined at their start")


if __name__ == "__main__":
    main()
