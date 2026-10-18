"""Serial native timing on predeclared held-out sizes, retaining failures."""

import argparse, hashlib, json, os, statistics, subprocess
from pathlib import Path

CASES = [
    ("orbit", 30),
    ("orbit", 48),
    ("separable", 96),
    ("separable", 192),
    ("restoration", 48),
    ("restoration", 96),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("builds", nargs="+")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cases", nargs="+")
    a = p.parse_args()
    cases = [(f, n) for f, n in CASES if not a.cases or f"{f}:{n}" in a.cases]
    builds = {
        name: str(Path(path).resolve())
        for name, path in (v.split("=", 1) for v in a.builds)
    }
    jobs = [
        (family, size, build, backend)
        for family, size in cases
        for build in builds
        for backend in ["active-set", "clarabel"]
    ]
    cpu = min(os.sched_getaffinity(0))
    rows = []
    for run in range(a.repeats + 1):
        order = jobs[run % len(jobs) :] + jobs[: run % len(jobs)]
        if run % 2:
            order = order[::-1]
        for family, size, build, backend in order:
            r = subprocess.run(
                [builds[build], family, str(size), "reuse", "0", backend],
                capture_output=True,
                text=True,
                timeout=120,
                preexec_fn=lambda: os.sched_setaffinity(0, {cpu}),
            )
            q = (
                json.loads(r.stdout)
                if r.returncode == 0
                else dict(
                    status="failed", returncode=r.returncode, stderr=r.stderr[-4000:]
                )
            )
            rows.append(
                dict(q, family=family, size=size, build=build, backend=backend, run=run)
            )
        print("holdout round", run, "done", flush=True)
    result = dict(
        cases=cases,
        repeats=a.repeats,
        warmups=1,
        cpu_affinity=cpu,
        binary_sha256={
            k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
            for k, v in builds.items()
        },
        records=rows,
        summary=[],
    )
    for family, size, build, backend in jobs:
        rr = [
            r
            for r in rows
            if r["run"]
            and (r["family"], r["size"], r["build"], r["backend"])
            == (family, size, build, backend)
        ]
        result["summary"].append(
            dict(
                family=family,
                size=size,
                build=build,
                backend=backend,
                passed=sum(r.get("status") != "failed" for r in rr),
                seconds=(
                    statistics.median(r["seconds"] for r in rr)
                    if all("seconds" in r for r in rr)
                    else None
                ),
            )
        )
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
