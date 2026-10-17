"""Block-curvature runs of the `performance` example, including failures.

    python3 scripts/benchmark_structured.py BINARY out.json

Records each (fixture, block width) once. A failed orbit run is recorded,
not raised: block curvature is an approximation that may not converge there.
The generated fixtures must converge.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNS = [("separable", 128, 2), ("separable", 128, 8), ("restoration", 64, 1),
        ("restoration", 64, 8), ("orbit", 20, 4), ("orbit", 40, 4)]


def main(binary: str, output: Path) -> None:
    records = []
    for family, size, block in RUNS:
        done = subprocess.run([str(Path(binary).resolve()), family, str(size), "reuse", str(block)],
                              capture_output=True, text=True, timeout=300)
        if done.returncode:
            records.append(dict(family=family, size=size, hessian_block_size=block,
                                accepted=False, error=done.stderr))
        else:
            records.append(dict(json.loads(done.stdout), accepted=True))
    output.write_text(json.dumps(records, indent=2) + "\n")
    for r in records:
        print(r["family"], r["size"], r["hessian_block_size"], r["accepted"], r.get("seconds"))
    failed = [r for r in records if r["family"] != "orbit" and not r["accepted"]]
    if failed:
        sys.exit(f"generated fixtures failed: {failed}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]))
