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

import json
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


def main() -> None:
    name, solver_names = sys.argv[1], sys.argv[2].split(",")
    channel = _channel()

    def emit(**data):
        channel.write(json.dumps(data) + "\n")
        channel.flush()

    from . import s2mpj, solvers
    from .adapter import S2mpjProblem
    from .check import check

    solvers.warm_up()
    began = time.perf_counter()
    problem = S2mpjProblem(s2mpj.load(name))
    emit(event="ready", setup_seconds=time.perf_counter() - began)
    for solver in solver_names:
        emit(event="start", solver=solver)
        record = solvers.run(solver, problem)
        verdict = (check(problem, record["x"], record.get("constraint_multipliers"),
                         record.get("variable_multipliers"))
                   if record.get("x") is not None else dict(accepted=False, reason="no point"))
        for key in ("x", "constraint_multipliers", "variable_multipliers"):
            record.pop(key, None)
        emit(event="result", problem=name, solver=solver, check=verdict, **record)


if __name__ == "__main__":
    main()
