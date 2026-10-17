"""Runs the benchmark: every selected problem with every solver, in isolated
worker processes.

    python -m bench.run selection.json results.jsonl
        [--solvers a,b,...] [--problems A,B,...] [--jobs 2] [--timeout 90]

Each problem gets a worker (bench/worker.py) that solves it with the solvers
in turn. A solve that exceeds `--timeout` seconds, or crashes the worker, is
recorded with status `timeout` or `crash`, and a new worker continues with
the remaining solvers. Results are appended as one JSON line per solve, so an
interrupted run resumes where it stopped: pairs already in the output are
skipped. An `environment` line records the versions and the machine.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import pathlib
import platform
import queue
import subprocess
import sys
import threading
import time

from . import s2mpj, solvers

ROOT = pathlib.Path(__file__).resolve().parents[1]
SETUP_TIMEOUT = 60.0


def environment() -> dict:
    import numpy
    import scipy

    import sqp_craft

    info = dict(event="environment", python=platform.python_version(), numpy=numpy.__version__,
                scipy=scipy.__version__, sqp_craft=sqp_craft.__version__,
                s2mpj_commit=s2mpj.COMMIT, machine=platform.machine(),
                cpus=os.cpu_count(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    try:
        import unopy

        info["uno"] = ".".join(str(getattr(unopy, f"VERSION_{p}")) for p in
                               ("MAJOR", "MINOR", "PATCH"))
    except ImportError:
        info["uno"] = None
    try:
        info["cpu"] = next(line.split(":", 1)[1].strip()
                           for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    try:
        info["sqp_craft_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["sqp_craft_commit"] = None
    return info


class Worker:
    """One worker process and a thread that queues its lines."""

    def __init__(self, name: str, solver_names: list[str]):
        self.lines: queue.Queue = queue.Queue()
        self.errors: collections.deque = collections.deque(maxlen=5)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "bench.worker", name, ",".join(solver_names)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, args=(self.process.stdout, True), daemon=True).start()
        threading.Thread(target=self._read, args=(self.process.stderr, False),
                         daemon=True).start()

    def _read(self, stream, results: bool) -> None:
        for line in stream:
            if results:
                self.lines.put(line)
            elif line.strip():
                self.errors.append(line.strip())
        if results:
            self.lines.put(None)

    def next(self, timeout: float):
        """The next event, `None` at the end of output, or `"timeout"`."""
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return "timeout"
        return None if line is None else json.loads(line)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


def failure(name: str, solver: str, status: str, message: str, seconds: float) -> dict:
    return dict(event="result", problem=name, solver=solver, status=status, claimed=False,
                check=dict(accepted=False, reason=status), iterations=None, functions=None,
                derivatives=None, hessians=None, seconds=seconds, message=message[:300])


def solve_problem(name: str, pending: list[str], timeout: float, emit) -> None:
    while pending:
        worker = Worker(name, pending)
        event = worker.next(SETUP_TIMEOUT)
        if not isinstance(event, dict) or event.get("event") != "ready":
            worker.stop()
            reason = "setup timeout" if event == "timeout" else "setup crash"
            for solver in pending:
                emit(failure(name, solver, "crash", f"{reason}: {' | '.join(worker.errors)}", 0.0))
            return
        while pending:
            event = worker.next(SETUP_TIMEOUT)
            if not isinstance(event, dict) or event.get("event") != "start":
                break
            solver, began = event["solver"], time.perf_counter()
            event = worker.next(timeout)
            if isinstance(event, dict) and event.get("event") == "result":
                emit(event)
                pending.remove(solver)
                continue
            elapsed = time.perf_counter() - began
            if event == "timeout":
                emit(failure(name, solver, "timeout", f"over {timeout:g} s", elapsed))
            else:
                worker.process.wait()
                emit(failure(name, solver, "crash",
                             f"exit {worker.process.returncode}: {' | '.join(worker.errors)}",
                             elapsed))
            pending.remove(solver)
            break
        worker.stop()


def done_pairs(path: pathlib.Path) -> set:
    pairs = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("event") == "result":
                    pairs.add((record["problem"], record["solver"]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("selection", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--solvers", default=",".join(list(solvers.SQP_VARIANTS) + solvers.COMPARATORS),
                        help="default: the sqp-craft variants and the comparators; "
                        "experiments (bench/solvers.py EXPERIMENTS) only when named")
    parser.add_argument("--problems", help="comma-separated names (default: the selection)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    names = (args.problems.split(",") if args.problems
             else [e["name"] for e in selection["problems"]])
    wanted = [s for s in args.solvers.split(",") if s]
    unknown = [s for s in wanted if s not in solvers.SOLVERS]
    if unknown:
        sys.exit(f"unknown solvers {unknown}; known: {', '.join(solvers.SOLVERS)}")
    missing = [s for s in wanted if not solvers.available(s)]
    if missing:
        print(f"skipping unavailable solvers: {', '.join(missing)}")
        wanted = [s for s in wanted if s not in missing]
    done = done_pairs(args.output)
    work = [(n, [s for s in wanted if (n, s) not in done]) for n in names]
    work = [(n, p) for n, p in work if p]
    lock = threading.Lock()
    with args.output.open("a") as out:
        out.write(json.dumps(environment()) + "\n")
        out.flush()
        progress = dict(problems=0, began=time.perf_counter())

        def emit(record: dict) -> None:
            with lock:
                out.write(json.dumps(record) + "\n")
                out.flush()

        def task(item) -> None:
            solve_problem(item[0], list(item[1]), args.timeout, emit)
            with lock:
                progress["problems"] += 1
                if progress["problems"] % 10 == 0 or progress["problems"] == len(work):
                    minutes = (time.perf_counter() - progress["began"]) / 60
                    print(f"{progress['problems']}/{len(work)} problems, {minutes:.1f} min",
                          flush=True)

        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(task, work))


if __name__ == "__main__":
    main()
