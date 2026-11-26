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
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
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

    def __init__(self, name: str, solver_names: list[str], *, evidence=False):
        self.lines: queue.Queue = queue.Queue()
        self.errors: collections.deque = collections.deque(maxlen=5)
        command = [sys.executable, "-m", "bench.worker", name, ",".join(solver_names)]
        if evidence:
            command.append("--evidence")
        self.process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def solve_problem(name: str, pending: list[str], timeout: float, emit, *,
                  evidence=False, dimensions=None, solve_limit=solvers.TIME_LIMIT) -> None:
    pending = list(pending)
    while pending:
        worker = Worker(name, pending, evidence=evidence)
        try:
            ready = worker.next(SETUP_TIMEOUT)
            if not isinstance(ready, dict) or ready.get("event") != "ready":
                reason = "setup timeout" if ready == "timeout" else "setup crash"
                for solver in pending:
                    emit(failure(name, solver, "crash", f"{reason}: {' | '.join(worker.errors)}", 0.0))
                return
            if dimensions is not None and ready["dimensions"] != dimensions:
                for solver in pending:
                    emit(failure(name, solver, "error",
                                 f"expected size {dimensions}, got {ready['dimensions']}", 0.0))
                return
            while pending:
                solver = pending.pop(0)
                start = worker.next(SETUP_TIMEOUT)
                if not isinstance(start, dict) or start.get("event") != "start" or start.get("solver") != solver:
                    emit(failure(name, solver, "crash", "worker stopped before the solve", 0.0))
                    break
                began = time.monotonic()
                limit = min(solve_limit, timeout) if evidence else timeout
                event = worker.next(limit)
                phase = "solve"
                if evidence and isinstance(event, dict) and event.get("event") == "solved":
                    if event["seconds"] > limit:
                        event = "timeout"
                    else:
                        phase = "verification watchdog"
                        remaining = timeout - (time.monotonic() - began)
                        event = worker.next(remaining) if remaining > 0 else "timeout"
                if isinstance(event, dict) and event.get("event") == "result" and event.get("solver") == solver:
                    emit(event)
                    continue
                elapsed = time.monotonic() - began
                if event == "timeout":
                    deadline = limit if phase == "solve" else timeout
                    emit(failure(name, solver, "timeout", f"{phase} exceeded {deadline:g} s", elapsed))
                else:
                    emit(failure(name, solver, "crash",
                                 f"{phase}: {' | '.join(worker.errors)}", elapsed))
                break
        finally:
            worker.stop()


def done_pairs(path: pathlib.Path, configuration=None) -> set:
    pairs = set()
    if path.exists():
        matched = False
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("event") == "environment":
                    if record.get("configuration_sha256") != configuration:
                        raise ValueError("output already contains a different configuration")
                    matched = True
                if record.get("event") == "result":
                    if configuration is not None and not matched:
                        raise ValueError("output contains results without a configuration")
                    pairs.add((record["problem"], record["solver"]))
    return pairs


def evidence_environment(info, selection, names, wanted, jobs, timeout):
    files = sorted((ROOT / "bench").glob("*.py")) + [ROOT / "Cargo.lock"]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in files}
    import sqp_craft
    import sqp_craft._native as native

    info["sqp_craft_python_sha256"] = hashlib.sha256(pathlib.Path(sqp_craft.__file__).read_bytes()).hexdigest()
    info.update(platform=platform.platform(), jobs=jobs, watchdog_seconds=timeout,
                solve_seconds_limit=solvers.TIME_LIMIT, max_iterations=solvers.MAX_ITERATIONS,
                sqp_craft_callback_limit=solvers.MAX_CALLBACKS, cases=names, solver_order=wanted,
                selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                source_sha256=source_hashes,
                sqp_craft_binary_sha256=hashlib.sha256(pathlib.Path(native.__file__).read_bytes()).hexdigest(),
                threads={name: os.environ.get(name) for name in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")})
    if any(name in solvers.pounce_adapter.MODES for name in wanted):
        try:
            info["pounce"] = version("pounce-solver")
        except PackageNotFoundError:
            info["pounce"] = None
        info.update(pounce_reference_commit=solvers.pounce_adapter.COMMIT,
                    pounce_wheel_sha256=os.environ.get("SQP_CRAFT_POUNCE_WHEEL_SHA256"))
    info["configuration_sha256"] = configuration_digest(info)


def configuration_digest(info):
    config = {key: value for key, value in info.items()
              if key not in ("started", "configuration_sha256")}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("selection", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--solvers",
                        default=",".join(list(solvers.SQP_VARIANTS) + solvers.COMPARATORS),
                        help="default: the sqp-craft variants and the comparators; "
                        "experiments (bench/solvers.py EXPERIMENTS) only when named")
    parser.add_argument("--problems", help="comma-separated names (default: the selection)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0, help="worker watchdog in seconds")
    parser.add_argument("--evidence", action="store_true",
                        help="record replay data and enforce the common solve deadline")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    names = (args.problems.split(",") if args.problems
             else [e["name"] for e in selection["problems"]])
    sizes = {e["name"]: [e["n"], e["m"]] for e in selection["problems"]} if args.evidence else {}
    if args.evidence:
        if selection.get("s2mpj_commit") != s2mpj.COMMIT:
            parser.error("selection S2MPJ pin does not match the harness")
        if (len(sizes) != len(selection["problems"]) or len(set(names)) != len(names)
                or not names or any(name not in sizes for name in names)):
            parser.error("expected a nonempty selection of unique named problems")
    if args.jobs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("jobs and watchdog must be positive")
    wanted = [s for s in args.solvers.split(",") if s]
    unknown = [s for s in wanted if s not in solvers.SOLVERS]
    if unknown:
        sys.exit(f"unknown solvers {unknown}; known: {', '.join(solvers.SOLVERS)}")
    missing = [s for s in wanted if not solvers.available(s)]
    if args.evidence and (not wanted or len(set(wanted)) != len(wanted)):
        parser.error("expected at least one distinct solver")
    if missing and not args.evidence:
        print(f"skipping unavailable solvers: {', '.join(missing)}")
        wanted = [s for s in wanted if s not in missing]
    info = environment()
    if args.evidence:
        evidence_environment(info, args.selection, names, wanted, args.jobs, args.timeout)
    try:
        done = done_pairs(args.output, configuration=info.get("configuration_sha256"))
    except ValueError as error:
        parser.error(f"{error}. Use a new output file.")
    work = [(n, [s for s in wanted if s not in missing and (n, s) not in done]) for n in names]
    work = [(n, p) for n, p in work if p]
    lock = threading.Lock()
    with args.output.open("a") as out:
        out.write(json.dumps(info) + "\n")
        out.flush()
        progress = dict(problems=0, began=time.perf_counter())

        def emit(record: dict) -> None:
            with lock:
                out.write(json.dumps(record) + "\n")
                out.flush()

        if args.evidence:
            for name in names:
                for solver in missing:
                    if (name, solver) not in done:
                        emit(failure(name, solver, "unsupported", "optional solver unavailable", 0.0))

        def task(item) -> None:
            name, pending = item
            solve_problem(name, pending, args.timeout, emit, evidence=args.evidence,
                          dimensions=sizes.get(name))
            with lock:
                progress["problems"] += 1
                if progress["problems"] % 10 == 0 or progress["problems"] == len(work):
                    minutes = (time.perf_counter() - progress["began"]) / 60
                    print(f"{progress['problems']}/{len(work)} problems, {minutes:.1f} min",
                          flush=True)

        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(task, work))
    if args.evidence and missing:
        sys.exit(f"unavailable solvers: {', '.join(missing)}")


if __name__ == "__main__":
    main()
