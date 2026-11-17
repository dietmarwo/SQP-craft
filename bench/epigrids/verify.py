"""Recheck recorded EPIGRIDS AC results against the original case files.

No optimizer runs. For every recorded candidate this recomputes the physical
check (balance, limits, cost) with PYPOWER's admittance matrices and the
first-order KKT residuals with SQP-craft's own checker, applies the frozen
acceptance gates and compares both with the record.

Usage:
    python bench/epigrids/verify.py --data /absolute/SyntheticElectricNetworkModels \
        --results bench/epigrids/results/2026-11-17/results.jsonl.gz

SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run as p  # noqa: E402

CASES = {
    "Winter12Generous": "Easy/Winter12Generous.m",
    "Fall116Generous": "Easy/Fall116Generous.m",
    "Winter68Generous": "Easy/Winter68Generous.m",
    "BaseCase": "BaseCase.m",
    "Spring13Tight": "Hard/Spring13Tight.m",
}
GATE = 1e-7  # balance and normalized limit violation, per unit
FIRST_ORDER = 1e-6  # stationarity and complementarity, original currency
COST = 1e-3  # relative cost above the accepted reference


def close(a, b):
    return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--results", type=pathlib.Path, required=True)
    a = ap.parse_args()
    with gzip.open(a.results, "rt") as f:
        lines = [json.loads(line) for line in f]
    cases = {r["case"]: r for r in lines if r["kind"] == "case"}
    accepted = executed = 0
    mismatches = []
    for record in (r for r in lines if r["kind"] == "ac"):
        case, start = record["case"], record["start"]
        executed += 1
        data, independent, identity = p.load(a.data / "New England/MatPower Cases" / CASES[case])
        if identity["sha256"] != cases[case]["import"]["sha256"]:
            raise SystemExit(f"{case}: the case file differs from the recorded one")
        model = p.Grid(data)
        reference = cases[case]["reference"]
        reference_ok = (
            reference["success"]
            and reference["checker"]["balance_pu"] <= GATE
            and reference["checker"]["limit_violation"] <= GATE
        )
        ok = False
        if record["candidate"] is not None and record["multipliers"] is not None:
            x = np.asarray(record["candidate"])
            physical = p.checker(model, x, independent)
            f, c = model.functions(x)
            gradient, jacobian = model.derivatives(x)
            evaluation = p.sqp.Evaluation(
                objective=f,
                constraints=c.tolist(),
                gradient=gradient.tolist(),
                jacobian_values=jacobian.tolist(),
                primal_violation=0.0,
                feasible_candidate=False,
            )
            kkt = p.sqp.kkt_residuals(model.spec(), x, evaluation, p.sqp.Multipliers(**record["multipliers"]))
            stationarity = kkt.stationarity * model.scale
            complementarity = kkt.complementarity * model.scale
            cost = (physical["cost"] - reference["objective"]) / max(1, abs(reference["objective"]))
            ok = bool(
                record["status"] == "converged"
                and reference_ok
                and physical["balance_pu"] <= GATE
                and physical["limit_violation"] <= GATE
                and stationarity <= FIRST_ORDER
                and complementarity <= FIRST_ORDER
                and record["dual_sign"] is not None
                and record["dual_sign"] <= 1e-8
                and kkt.dual_sign <= 1e-8
                and cost <= COST
            )
            recomputed = {
                "stationarity_currency": stationarity,
                "complementarity_currency": complementarity,
                "cost_relative_to_reference": cost,
                **{f"checker.{k}": v for k, v in physical.items()},
            }
            recorded = {
                "stationarity_currency": record["stationarity_currency"],
                "complementarity_currency": record["complementarity_currency"],
                "cost_relative_to_reference": record["cost_relative_to_reference"],
                **{f"checker.{k}": v for k, v in record["checker"].items()},
            }
            mismatches += [(case, start, k) for k in recomputed if not close(recomputed[k], recorded[k])]
        if ok != record["accepted"]:
            mismatches.append((case, start, "accepted"))
        accepted += ok
        print(f"{case:17s} {start:21s} {record['status']:20s} accepted={ok}")
    print(f"{accepted}/{executed} accepted; {len(mismatches)} differences from the record")
    for m in mismatches:
        print("differs:", *m)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
