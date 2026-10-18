"""Fresh-process thread-budget adapter around unchanged mathematical generators."""

import argparse
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--kind", choices=["stress", "independent", "variants"], required=True)
p.add_argument("--threads", type=int, choices=[1, 8], required=True)
a, extra = p.parse_known_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "examples"))
import sqp_craft
import compare_baseline as cb
import stress_random as sr

native = importlib.import_module("sqp_craft._native")
original_solve = sqp_craft.solve
original_evaluate = cb.evaluate
journal = []
checks = {}


def forced_solve(*args, **kwargs):
    options = kwargs.get("options") or sqp_craft.Options()
    kwargs["options"] = dataclasses.replace(options, threads=a.threads)
    index = len(journal)
    journal.append(dict(index=index, status="exception"))
    result = original_solve(*args, **kwargs)
    journal[index] = dict(
        index=index,
        status=result.status,
        iterations=result.iterations,
        objective=(
            result.evaluation.objective if result.evaluation is not None else None
        ),
        candidate=result.candidate,
        functions=result.counts.functions,
        derivatives=result.counts.derivatives,
        backend=result.report.qp_backend,
    )
    return result


def evaluated(*args, **kwargs):
    answer = original_evaluate(*args, **kwargs)
    checks[len(journal) - 1] = answer
    return answer


sqp_craft.solve = forced_solve
cb.evaluate = evaluated
scripts = {
    "stress": "validate_performance_stress.py",
    "independent": "performance30_independent.py",
    "variants": "compare_variants.py",
}
payload = a.output.with_suffix(".payload.json")
sys.argv = [
    scripts[a.kind],
    *(
        [str(payload), *extra]
        if a.kind != "variants"
        else [*extra, "--json", str(payload)]
    ),
]
try:
    runpy.run_path(str(root / "scripts" / scripts[a.kind]), run_name="__main__")
finally:
    if payload.exists():
        for row in journal:
            check = checks.get(row["index"])
            row["check"] = check
            row["solver_certificate_accepted"] = bool(
                check and sr.first_order_ok(check) and check["certificate"] == "solver"
            )
        envelope = dict(
            version=sqp_craft.__version__,
            threads=a.threads,
            kind=a.kind,
            extension_sha256=hashlib.sha256(
                Path(native.__file__).read_bytes()
            ).hexdigest(),
            observed_solver_calls=len(journal),
            journal=journal,
            payload=json.loads(payload.read_text()),
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
        payload.unlink()
