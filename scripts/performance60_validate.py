"""Three-way independent numerical validation. Run serially after all timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-python", type=Path, required=True)
    p.add_argument(
        "--output", type=Path, default=Path("docs/performance-60/data/stress/full")
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--current-version", default="0.16.0")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    d = a.output.resolve()
    d.mkdir(parents=True, exist_ok=True)
    logs = d / "logs"
    logs.mkdir(exist_ok=True)
    count = 100 if a.quick else 300
    seeds = [17] if a.quick else [1, 2, 3, 4, 5, 2026]
    groups = {}
    for backend in ["active-set", "clarabel"]:
        for seed in seeds:
            groups[f"{backend}-seed{seed}"] = (
                "stress",
                ["--seed", str(seed), "--count", str(count), "--qp-solver", backend],
                count * 3,
            )
    groups["elastic-seed23"] = (
        "stress",
        [
            "--seed",
            "23",
            "--count",
            str(count),
            "--qp-solver",
            "active-set",
            "--elastic",
        ],
        count * 3,
    )
    groups["independent-development"] = ("independent", ["--development"], 8)
    groups["independent-heldout"] = ("independent", [], 12)
    if not a.quick:
        groups["variants"] = (
            "variants",
            [
                "--variants",
                "default,soc,augmented,scale-jacobian",
                "--families",
                "baseline,maratos,linear,mpcc",
                "--qp-solver",
                "active-set",
            ],
            5652,
        )
    hashes = json.loads((root / "docs/performance-60/data/inputs.json").read_text())[
        "sha256"
    ]
    for name, sha in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    labels = {"baseline": 1, "serial": 1, "parallel": 8}
    manifest = dict(
        quick=a.quick,
        baseline_version="0.15.0",
        current_version=a.current_version,
        groups={k: dict(kind=v[0], expected_calls=v[2]) for k, v in groups.items()},
        labels=labels,
        expected_calls_per_variant=sum(v[2] for v in groups.values()),
        callback_sha256=hashes,
    )
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for group, (kind, extra, _) in groups.items():
        for label, width in labels.items():
            env = dict(
                os.environ,
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
            )
            if label == "baseline":
                env["PYTHONPATH"] = str(a.baseline_python.resolve())
            else:
                env.pop("PYTHONPATH", None)
            name = f"{label}-{group}"
            with (logs / (name + ".log")).open("w") as out:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/performance60_worker.py",
                        str(d / (name + ".json")),
                        "--kind",
                        kind,
                        "--threads",
                        str(width),
                        *extra,
                    ],
                    cwd=root,
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(name, "passed", flush=True)
    print("all three-way workers passed", flush=True)


if __name__ == "__main__":
    main()
