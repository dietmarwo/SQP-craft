"""Generate the frozen multiple-shooting campaign for experiment-craft."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ARCS = (7, 13, 26, 51, 101)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new directory for the manifest and staged sources")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python with sqp-craft >= 0.18.0, pykep-rust >= 0.1.6, NumPy and SciPy installed")
    args = parser.parse_args()
    args.directory.mkdir(parents=True)
    root = Path(__file__).resolve().parent
    sources = ["__init__.py", "trajectory_scale.py", "solvers.py"]
    (args.directory / "bench").mkdir()
    for name in sources:
        shutil.copyfile(root / name, args.directory / "bench" / name)
    inputs = [dict(path=f"bench/{name}", file=f"bench/{name}") for name in sources]

    def step(arc, method, seed):
        return dict(
            id=f"a{arc}-{method}-{seed}",
            job=dict(
                schema_version="execution.command/0.1",
                argv=[str(args.python.absolute()), "-m", "bench.trajectory_scale", "--arcs", str(arc),
                      "--method", method, "--scaling", "none", "--seed", seed, "--output", "result.json"],
                env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                resources={"cpu_slots": 1},
                limits={"wall_seconds": 40, "grace_seconds": 2},
                replay="never",
                inputs=inputs,
                outputs=[dict(name="result", path="result.json", required=True, max_bytes=2_000_000)],
            ),
        )

    steps = [step(a, method, "lambert") for a in ARCS for method in ("derivatives", "partitioned")]
    steps.extend(step(a, "partitioned", "offset") for a in (51, 101))
    manifest = dict(schema_version="experiment.campaign/0.1",
                    name="SQP5 multiple-shooting scale study", allow_commands=True,
                    budget=dict(max_jobs=len(steps), max_wall_seconds=650,
                                max_reserved_slot_seconds=42*len(steps)),
                    steps=steps)
    path = args.directory / "campaign.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
