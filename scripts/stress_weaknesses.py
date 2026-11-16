"""Per-case records for the fixed 3,000-run stress panel, with independent checks."""

import argparse
import dataclasses
import gzip
import json
import math
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import compare_baseline as cb
import stress_random as sr
import stress_mpcc as mp
from sqp_craft import SqpOptions, solve


def clean(data):
    if isinstance(data, dict):
        return {k: clean(v) for k, v in data.items() if "seconds" not in k}
    if isinstance(data, list):
        return [clean(v) for v in data]
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--variant", choices=["off", "step", "safeguard", "safeguard10"], default="off")
    a = p.parse_args()
    fields = {
        "off": {},
        "step": dict(major_step_limit=2.0),
        "safeguard": dict(major_step_limit=2.0, initial_curvature_scaling=True),
        "safeguard10": dict(major_step_limit=10.0, initial_curvature_scaling=True),
    }[a.variant]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    false = 0
    count = 0
    with gzip.open(a.output, "wt") as out:
        for module, seeds in [(sr, range(1, 5)), (mp, range(1, 3))]:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                for k in range(300):
                    spec, model, start = module.problem(rng)
                    case = cb.Case(module.__name__, spec, model, start, 0.0, math.inf)
                    for restoration in [10, 0] if module is sr else [10]:
                        options = SqpOptions(
                            max_iterations=300,
                            qp_solver="auto",
                            elastic_mode=None,
                            max_restoration_steps=restoration,
                            **fields,
                        )
                        r = solve(spec, model(), start, method="sqp", method_options=options)
                        check = (
                            cb.evaluate(case, r.candidate, r.multipliers)
                            if r.candidate is not None
                            else None
                        )
                        ok = sr.first_order_ok(check)
                        record = dict(
                            family=module.__name__,
                            seed=seed,
                            index=k,
                            restoration=restoration,
                            status=r.status,
                            candidate=r.candidate,
                            counts=dataclasses.asdict(r.counts),
                            iterations=r.iterations,
                            evaluation=dataclasses.asdict(r.evaluation) if r.evaluation else None,
                            multipliers=dataclasses.asdict(r.multipliers)
                            if r.multipliers
                            else None,
                            stationarity=r.stationarity,
                            complementarity=r.complementarity,
                            dual_sign=r.dual_sign,
                            check=check,
                            accepted=ok,
                            message=r.message,
                            report=clean(dataclasses.asdict(r.report)),
                        )
                        out.write(
                            json.dumps(
                                record, default=lambda v: v.item() if hasattr(v, "item") else str(v)
                            )
                            + "\n"
                        )
                        false += int(r.status == "converged" and not ok)
                        count += 1
                print(module.__name__, seed, count, "false", false, flush=True)
    assert count == 3000 and false == 0, (count, false)


if __name__ == "__main__":
    main()
