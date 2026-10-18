"""Serial paired eight-CPU-budget benchmarks; each child may use several workers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("builds", nargs="+", help="LABEL=performance_parallel executable")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--full", action="store_true")
    p.add_argument("--baseline", action="store_true", help="measure only supported 0.15 modes")
    p.add_argument("--qp-only", action="store_true")
    p.add_argument("--case", action="append", help="family:size:backend:inner:outer:jobs")
    a = p.parse_args()
    builds = dict(v.split("=", 1) for v in a.builds)
    # inner threads * outer workers never exceeds eight.
    cases = [("orbit", 40, "active-set", 1, 1, 1), ("orbit", 20, "active-set", 1, 8, 16)]
    if a.baseline:
        cases += [("orbit", 20, "active-set", 1, 1, 16)]
    else:
        cases += [("orbit", 40, "active-set", 8, 1, 1), ("separable", 128, "active-set", 8, 1, 1)]
    if a.full:
        cases = []
        for backend in ["active-set", "clarabel"]:
            for family, size in [("orbit", 20), ("orbit", 40), ("separable", 128), ("restoration", 64)]:
                for width in ([1] if a.baseline else [1, 2, 4, 8]):
                    cases.append((family, size, backend, width, 1, 1))
            for family, size in [("orbit", 20), ("separable", 128)]:
                for inner, outer in ([(1, 1), (1, 8)] if a.baseline else [(1, 1), (1, 8), (2, 4), (4, 2), (8, 1)]):
                    cases.append((family, size, backend, inner, outer, 32))
    if a.qp_only:
        cases = [("qp", n, "active-set", w, 1, 1) for n in [128, 256, 512, 1024] for w in [1, 2, 4, 8]]
    if a.case:
        cases = [(v[0],int(v[1]),v[2],int(v[3]),int(v[4]),int(v[5])) for value in a.case for v in [value.split(":")]]
    records = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for repeat in range(a.repeats + 1):
        for ci, case in enumerate(cases):
            order = list(builds)
            if (repeat + ci) % 2:
                order.reverse()
            for label in order:
                family, size, backend, inner, outer, jobs = case
                command = [builds[label], family, str(size), "reuse", "0", backend, str(inner), str(outer), str(jobs)]
                if a.qp_only:
                    command = [builds[label], str(size), str(inner)]
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                began = time.perf_counter()
                child = subprocess.run(command, capture_output=True, text=True, env=env)
                wall = time.perf_counter() - began
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                rows = []
                if not child.returncode:
                    rows = [json.loads(line) for line in child.stdout.splitlines() if line.startswith("{")]
                    assert len(rows) == jobs + 1 and rows[-1]["jobs"] == jobs
                record = dict(build=label, repeat=repeat, case=list(case), command=command,
                              returncode=child.returncode, process_seconds=wall,
                              cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                              results=rows, stderr=child.stderr)
                records.append(record)
                # Preserve unsupported old checkpoint outcomes without treating them as wins.
                if child.returncode and "single-threaded" not in child.stderr and "parallel threads" not in child.stderr:
                    raise RuntimeError(record)
    summary = []
    for case in cases:
        for label in builds:
            runs = [r for r in records if r["repeat"] and r["build"] == label and tuple(r["case"]) == case]
            passed = [r for r in runs if not r["returncode"]]
            row = dict(build=label, case=list(case), passed=len(passed), samples=len(runs))
            if passed:
                row.update(seconds=statistics.median(r["results"][-1]["seconds"] for r in passed),
                           cpu_seconds=statistics.median(r["cpu_seconds"] for r in passed),
                           objective=passed[0]["results"][0]["objective"],
                           iterations=passed[0]["results"][0]["iterations"])
            summary.append(row)
    result = dict(repeats=a.repeats, cases=cases,
                  environment=dict(affinity=sorted(os.sched_getaffinity(0)), cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip()),
                  binaries={k: dict(path=v, sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest()) for k, v in builds.items()},
                  records=records, summary=summary)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
