"""Serial paired benchmarks of explicit eight-CPU-budget native work."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time


def validate_rows(rows, case):
    family, size, _, _, outer, jobs = case
    if len(rows) != jobs + 1 or rows[-1].get("jobs") != jobs:
        raise ValueError("missing ordered job results")
    if rows[-1].get("workers", outer) != outer:
        raise ValueError("wrong outer worker count")
    for index, row in enumerate(rows[:-1]):
        expected_family, expected_size = family, size
        if family == "mixed":
            expected_family, expected_size = "orbit", 40 if index % 8 == 0 else 20
        if row.get("family") != expected_family or row.get("size") != expected_size:
            raise ValueError("wrong problem or result order")
        for key in ["seconds", "objective", "primal", "stationarity"]:
            if key not in row or not math.isfinite(row[key]):
                raise ValueError("missing/nonfinite numeric result: " + key)
        if row["seconds"] <= 0:
            raise ValueError("nonpositive measured work")
        for key, tolerance in [
            ("primal", 1e-7),
            ("stationarity", 1e-6),
            ("complementarity", 1e-6),
            ("dual_sign", 1e-8),
        ]:
            # The frozen 0.15 binary asserts all four, but prints only two.
            if key in row and not (
                math.isfinite(row[key]) and 0 <= row[key] <= tolerance
            ):
                raise ValueError("rejected original-unit residual: " + key)
        if expected_family == "orbit" and expected_size in [20, 40]:
            reference = 1.52206 if expected_size == 20 else 1.52449
            if abs(row["objective"] + reference) >= 5e-5:
                raise ValueError("rejected original reference objective")
        elif expected_family in ["separable", "restoration", "propagation"]:
            if row["objective"] >= 1e-9:
                raise ValueError("rejected known-zero objective")
    if not (math.isfinite(rows[-1]["seconds"]) and rows[-1]["seconds"] > 0):
        raise ValueError("invalid total work time")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("builds", nargs="+", help="LABEL=performance_parallel executable")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--full", action="store_true")
    p.add_argument("--holdout", action="store_true")
    p.add_argument(
        "--baseline", action="store_true", help="only modes supported by frozen0.15"
    )
    p.add_argument("--batch-jobs", type=int, default=32)
    p.add_argument("--qp-only", action="store_true")
    p.add_argument("--qp-sizes", default="128,256,512,1024")
    p.add_argument(
        "--case", action="append", help="family:size:backend:inner:outer:jobs"
    )
    a = p.parse_args()
    if a.repeats < 1 or a.batch_jobs < 1:
        p.error("positive measured repeats and batch jobs required")
    builds = dict(v.split("=", 1) for v in a.builds)
    cases = [
        ("orbit", 40, "active-set", 1, 1, 1),
        ("orbit", 20, "active-set", 1, 8, 16),
    ]
    cases += (
        [("orbit", 20, "active-set", 1, 1, 16)]
        if a.baseline
        else [
            ("orbit", 40, "active-set", 8, 1, 1),
            ("separable", 128, "active-set", 8, 1, 1),
        ]
    )
    if a.full or a.holdout:
        cases = []
        families = (
            [
                ("orbit", 30),
                ("orbit", 48),
                ("separable", 96),
                ("separable", 192),
                ("restoration", 48),
                ("restoration", 96),
            ]
            if a.holdout
            else [("orbit", 20), ("orbit", 40), ("separable", 128), ("restoration", 64)]
        )
        if not a.baseline:
            families += (
                [("propagation", 48), ("propagation", 96)]
                if a.holdout
                else [("separable", 256), ("propagation", 32)]
            )
        widths = [1] if a.baseline else ([1, 8] if a.holdout else [1, 2, 4, 8])
        for backend in ["active-set", "clarabel"]:
            for family, size in families:
                for width in widths:
                    cases.append((family, size, backend, width, 1, 1))
            if not a.holdout:
                splits = (
                    [(1, 1), (1, 8)]
                    if a.baseline
                    else [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (4, 2), (8, 1)]
                )
                for family, size in [("orbit", 20), ("separable", 128)]:
                    for inner, outer in splits:
                        cases.append(
                            (family, size, backend, inner, outer, a.batch_jobs)
                        )
        if not a.baseline and not a.holdout:
            cases += [
                ("mixed", 20, "active-set", 1, outer, a.batch_jobs) for outer in [1, 8]
            ]
    if a.qp_only:
        cases = [
            ("qp", n, "active-set", w, 1, 1)
            for n in map(int, a.qp_sizes.split(","))
            for w in [1, 2, 4, 8]
        ]
    if a.case:
        cases = [
            (v[0], int(v[1]), v[2], int(v[3]), int(v[4]), int(v[5]))
            for value in a.case
            for v in [value.split(":")]
        ]
    for _, size, _, inner, outer, jobs in cases:
        if min(size, inner, outer, jobs) < 1 or inner * outer > 8:
            p.error(
                "positive sizes/counts and outer workers times inner threads <=8 required"
            )
    records = []
    affinity = sorted(os.sched_getaffinity(0))[:8]
    environment = dict(
        parent_affinity=sorted(os.sched_getaffinity(0)),
        child_affinity=affinity,
        cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
    )
    binaries = {
        k: dict(path=v, sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest())
        for k, v in builds.items()
    }
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    progress = a.output.with_suffix(".progress.json")
    for repeat in range(a.repeats + 1):
        for ci, case in enumerate(cases):
            order = list(builds)
            if (repeat + ci) % 2:
                order.reverse()
            for label in order:
                family, size, backend, inner, outer, jobs = case
                command = (
                    [builds[label], str(size), str(inner)]
                    if a.qp_only
                    else [
                        builds[label],
                        family,
                        str(size),
                        "reuse",
                        "0",
                        backend,
                        str(inner),
                        str(outer),
                        str(jobs),
                    ]
                )
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                began = time.perf_counter()
                error = None
                try:
                    child = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=180,
                        preexec_fn=lambda: os.sched_setaffinity(0, affinity),
                    )
                    returncode, stdout, stderr = (
                        child.returncode,
                        child.stdout,
                        child.stderr,
                    )
                except subprocess.TimeoutExpired as exc:
                    returncode, stdout, stderr = -9, "", str(exc)
                wall = time.perf_counter() - began
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                rows = []
                try:
                    if returncode:
                        raise ValueError("benchmark process failed")
                    rows = [
                        json.loads(line)
                        for line in stdout.splitlines()
                        if line.startswith("{")
                    ]
                    validate_rows(rows, case)
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                record = dict(
                    build=label,
                    repeat=repeat,
                    case=list(case),
                    command=command,
                    returncode=returncode,
                    process_seconds=wall,
                    cpu_seconds=after.ru_utime
                    + after.ru_stime
                    - before.ru_utime
                    - before.ru_stime,
                    results=rows,
                    stderr=stderr,
                    error=error,
                )
                records.append(record)
                progress.write_text(
                    json.dumps(
                        dict(
                            completed=len(records),
                            total=(a.repeats + 1) * len(cases) * len(builds),
                            repeat=repeat,
                            case=case,
                            build=label,
                            error=error,
                        )
                    )
                )
                if error:
                    a.output.write_text(
                        json.dumps(
                            dict(
                                status="failed",
                                error=error,
                                records=records,
                                binaries=binaries,
                                environment=environment,
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    raise RuntimeError(
                        f"{label} {case}: {error}; retained evidence in {a.output}"
                    )
    summary = []
    for case in cases:
        for label in builds:
            runs = [
                r
                for r in records
                if r["repeat"] and r["build"] == label and tuple(r["case"]) == case
            ]
            row = dict(
                build=label,
                case=list(case),
                passed=len(runs),
                samples=len(runs),
                seconds=statistics.median(r["results"][-1]["seconds"] for r in runs),
                solver_seconds_per_job=statistics.median(
                    job["seconds"] for r in runs for job in r["results"][:-1]
                ),
                process_seconds=statistics.median(r["process_seconds"] for r in runs),
                cpu_seconds=statistics.median(r["cpu_seconds"] for r in runs),
                objective=runs[0]["results"][0]["objective"],
                iterations=runs[0]["results"][0]["iterations"],
            )
            summary.append(row)
    result = dict(
        status="passed",
        repeats=a.repeats,
        cases=cases,
        environment=environment,
        binaries=binaries,
        records=records,
        summary=summary,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    progress.unlink(missing_ok=True)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
