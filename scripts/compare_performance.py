"""Paired native benchmark of several builds of the `performance` example.

    cargo build --release --locked -p sqp-craft-solver --example performance
    python3 scripts/compare_performance.py out.json base=/path/a new=/path/b \\
        [--blocks new] [--repeats 5]

Each LABEL=BINARY is run on the fixed-callback fixtures with reused and cold
QP workspaces; builds named with --blocks also run their block mode (width 4
on the orbit, 2 on the separable and 1 on the restoration fixture). Every
call is a fresh process. The first round is a warmup; rounds rotate and
alternate direction so drift on the host affects every build alike. The
example asserts convergence, original-unit KKT residuals and the reference
objectives itself. Nothing here asserts a timing.

Wall time includes the Clarabel engine, whose speed shifts by a few percent
with code layout between builds even on identical inputs; compare the BFGS
and conversion phases for changes outside the engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path

CASES = [("orbit", 20), ("orbit", 40), ("separable", 128), ("restoration", 64)]
BLOCK_WIDTH = {"orbit": 4, "separable": 2, "restoration": 1}
METRICS = ["seconds", "iterations", "qp_calls", "qp_iterations", "bfgs_seconds",
           "conversion_seconds", "engine_seconds", "setups", "updates", "objective"]


def cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("builds", nargs="+", metavar="LABEL=BINARY")
    parser.add_argument("--blocks", nargs="*", default=[], metavar="LABEL",
                        help="builds that also run their block mode")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    builds = dict(item.split("=", 1) for item in args.builds)
    unknown = set(args.blocks) - builds.keys()
    if unknown:
        parser.error(f"--blocks names unknown builds: {sorted(unknown)}")

    variants = [(label, mode, False) for label in builds for mode in ("reuse", "cold")]
    variants += [(label, "reuse", True) for label in args.blocks]
    jobs = [(family, size, *variant) for family, size in CASES for variant in variants]
    result = dict(cpu=cpu(), platform=platform.platform(), repeats=args.repeats, warmups=1,
                  binary_sha256={label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for label, path in builds.items()},
                  records=[])
    for run in range(args.repeats + 1):
        shift = run % len(jobs)
        order = jobs[shift:] + jobs[:shift]
        if run % 2:
            order.reverse()
        for family, size, label, mode, blocked in order:
            width = BLOCK_WIDTH[family] if blocked else 0
            command = [str(Path(builds[label]).resolve()), family, str(size), mode, str(width)]
            done = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if done.returncode:
                raise RuntimeError(f"{label} {family} {size} {mode}: {done.stderr[-2000:]}")
            record = json.loads(done.stdout)
            record.update(build=label, mode=mode + ("+blocks" if blocked else ""), run=run)
            result["records"].append(record)
        print(f"round {run} of {args.repeats} done ({'warmup' if run == 0 else 'measured'})",
              flush=True)

    result["summary"] = []
    for family, size in CASES:
        for label, mode, blocked in variants:
            name = mode + ("+blocks" if blocked else "")
            rows = [r for r in result["records"] if r["run"] > 0 and r["family"] == family
                    and r["size"] == size and r["build"] == label and r["mode"] == name]
            summary = dict(family=family, size=size, build=label, mode=name)
            summary.update({key: statistics.median(r[key] for r in rows) for key in METRICS})
            result["summary"].append(summary)
            print(f"{family:12s}{size:4d}  {label + ' ' + name:28s}"
                  f"{summary['seconds'] * 1e3:10.2f} ms  BFGS {summary['bfgs_seconds'] * 1e3:7.2f}"
                  f"  conversion {summary['conversion_seconds'] * 1e3:7.2f}"
                  f"  engine {summary['engine_seconds'] * 1e3:9.2f}  steps {summary['iterations']:.0f}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
