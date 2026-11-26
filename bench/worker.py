"""Solves one problem with several solvers in turn, reporting one JSON line per
event on a private channel (the original standard output):

    {"event": "ready", "setup_seconds": ...}
    {"event": "start", "solver": ...}
    {"event": "result", ...record...}

    python -m bench.worker NAME SOLVER[,SOLVER...]

Anything the solvers print, from Python or from C, goes to the null device,
so it cannot corrupt the channel. The parent (bench/run.py) enforces the time
limit per solve and restarts the worker after a timeout or a crash.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time


def _channel():
    """A text stream on the original standard output, with fd 1 sent to the
    null device."""
    fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = open(os.devnull, "w")
    return os.fdopen(fd, "w", buffering=1)


def objective_check(reported, verified):
    if reported is None or verified is None:
        return dict(delta=None, mismatch=None)
    if not (math.isfinite(reported) and math.isfinite(verified)):
        return dict(delta=None, mismatch=True)
    delta = reported - verified
    return dict(delta=delta, mismatch=abs(delta) > 1e-8 * max(1.0, abs(verified)))


def verify(problem, record, *, evidence=True):
    from .check import check

    if record["x"] is None:
        return dict(accepted=False, reason="no point")
    raw = record.get("raw_multipliers")
    sides = (raw["mult_x_L"], raw["mult_x_U"]) if raw is not None else None
    return check(problem, record["x"], record.get("constraint_multipliers"),
                 record.get("variable_multipliers"), include_solver_certificate=evidence,
                 bound_duals=sides)


def main() -> None:
    name, solver_names = sys.argv[1], sys.argv[2].split(",")
    evidence = "--evidence" in sys.argv[3:]
    channel = _channel()

    def emit(**data):
        channel.write(json.dumps(data) + "\n")
        channel.flush()

    from . import s2mpj, solvers
    from .adapter import S2mpjProblem

    began = time.perf_counter()
    solvers.warm_up(solver_names)
    problem = S2mpjProblem(s2mpj.load(name))
    setup = time.perf_counter() - began
    emit(event="ready", setup_seconds=setup, dimensions=[problem.n, problem.m])
    if evidence:
        start_sha = hashlib.sha256(problem.x0.astype("<f8").tobytes()).hexdigest()
    for solver in solver_names:
        emit(event="start", solver=solver)
        record = solvers.run(solver, problem, evidence=evidence)
        if evidence:
            emit(event="solved", seconds=record["seconds"])
        began = time.perf_counter()
        verdict = verify(problem, record, evidence=evidence)
        if evidence:
            record.update(check_seconds=time.perf_counter() - began, setup_seconds=setup,
                          start_sha256=start_sha, start_clipped=problem.start_clipped,
                          objective_check=objective_check(
                              record.get("reported_objective"), verdict.get("objective")))
        else:
            for key in ("x", "constraint_multipliers", "variable_multipliers", "raw_multipliers"):
                record.pop(key, None)
        emit(event="result", problem=name, solver=solver, check=verdict, **record)


if __name__ == "__main__":
    main()
