import subprocess, os, json, sys, argparse
from pathlib import Path

R = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(
    description="Serial paired NLP validation; run after all timing."
)
p.add_argument("--baseline-python", type=Path, required=True)
p.add_argument("--output", type=Path, default=R / "docs/performance-30/data/stress")
a = p.parse_args()
D = a.output.resolve()
D.mkdir(parents=True, exist_ok=True)
V = D / "logs"
V.mkdir(exist_ok=True)


def run(label, args, filename):
    env = os.environ.copy()
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[name] = "1"
    if label == "baseline":
        env["PYTHONPATH"] = str(a.baseline_python.resolve())
    else:
        env.pop("PYTHONPATH", None)
    with (V / filename).open("w") as log:
        subprocess.run(
            [sys.executable, *args],
            cwd=R,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(label, filename, "passed", flush=True)


for backend in ["active-set", "clarabel"]:
    for seed in [1, 2, 3, 4, 5, 2026]:
        for label in ["baseline", "final"]:
            name = f"{label}-{backend}-seed{seed}"
            run(
                label,
                [
                    "scripts/validate_performance_stress.py",
                    str(D / (name + ".json")),
                    "--seed",
                    str(seed),
                    "--count",
                    "300",
                    "--qp-solver",
                    backend,
                ],
                name + ".log",
            )
for label in ["baseline", "final"]:
    name = label + "-elastic-seed23"
    run(
        label,
        [
            "scripts/validate_performance_stress.py",
            str(D / (name + ".json")),
            "--seed",
            "23",
            "--count",
            "300",
            "--qp-solver",
            "active-set",
            "--elastic",
        ],
        name + ".log",
    )
    for group, extra in [("development", ["--development"]), ("heldout", [])]:
        name = label + "-independent-" + group
        run(
            label,
            ["scripts/performance30_independent.py", str(D / (name + ".json")), *extra],
            name + ".log",
        )
    name = label + "-variants"
    run(
        label,
        [
            "scripts/compare_variants.py",
            "--variants",
            "default,soc,augmented,scale-jacobian",
            "--families",
            "baseline,maratos,linear,mpcc",
            "--qp-solver",
            "active-set",
            "--json",
            str(D / (name + ".json")),
        ],
        name + ".log",
    )
print("all final independent validation workers passed", flush=True)
