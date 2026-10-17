"""Median native timings of one build of the `performance` example.

    python3 scripts/performance_iteration.py BINARY out.json [--repeats 3]

One warmup and REPEATS measured rounds on the fixed-callback fixtures with a
reused QP workspace, alternating direction. The example asserts convergence,
original-unit KKT residuals and reference objectives; no timing is asserted.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path

CASES = [("orbit", 20), ("orbit", 40), ("separable", 128), ("restoration", 64)]


def measure(binary: str, repeats: int = 3) -> dict:
    records = []
    for run in range(repeats + 1):
        for family, size in CASES if run % 2 == 0 else list(reversed(CASES)):
            done = subprocess.run([str(Path(binary).resolve()), family, str(size), "reuse"],
                                  capture_output=True, text=True, timeout=300)
            if done.returncode:
                raise RuntimeError(done.stderr)
            records.append(dict(json.loads(done.stdout), run=run))
    summary = []
    for family, size in CASES:
        rows = [r for r in records if r["run"] and r["family"] == family and r["size"] == size]
        numeric = [k for k, v in rows[0].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("size", "run")]
        summary.append(dict(family=family, size=size,
                            **{k: statistics.median(r[k] for r in rows) for k in numeric}))
    return dict(platform=platform.platform(), repeats=repeats, records=records, summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary")
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = measure(args.binary, args.repeats)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["summary"]:
        print(f"{row['family']:12s}{row['size']:4d}{row['seconds'] * 1e3:10.2f} ms  "
              f"steps {row['iterations']:.0f}")
