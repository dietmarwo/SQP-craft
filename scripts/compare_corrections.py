"""Paired runs with and without the second-order correction.

    python3 scripts/compare_corrections.py [--qp-solver clarabel|active-set]

Three families, each solved twice (`second_order_correction` off and on):
the random stress problems of examples/stress_random.py (seeds 1–5), the
complementarity problems of examples/stress_mpcc.py (seeds 1–2), and 200
starts of the Maratos example of Nocedal & Wright (Example 15.4),
min 2(x₁² + x₂² − 1) − x₁ subject to x₁² + x₂² = 1, at random angles and
radii 0.5–2. Every convergence claim is checked by the independent
evaluator of compare_baseline.py; the script fails on a false claim.
Prints statuses, total iterations and callbacks, and the totals over the
problems that converge both ways.
"""
from __future__ import annotations

import argparse
import collections
import math
import pathlib
import sys

import numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "examples"))
import compare_baseline as cb  # noqa: E402
import stress_mpcc as sm  # noqa: E402
import stress_random as sr  # noqa: E402

from sqp_craft import ProblemSpec, SqpOptions, solve  # noqa: E402


class Maratos:
    def functions(self, x):
        return 2 * (x[0] ** 2 + x[1] ** 2 - 1) - x[0], [x[0] ** 2 + x[1] ** 2]

    def derivatives(self, x):
        return [4 * x[0] - 1, 4 * x[1]], [2 * x[0], 2 * x[1]]


def maratos_problems():
    spec = ProblemSpec(lower=[-math.inf] * 2, upper=[math.inf] * 2,
                       constraint_lower=[1.0], constraint_upper=[1.0],
                       jacobian_indptr=[0, 2], jacobian_indices=[0, 1])
    rng = np.random.default_rng(3)
    for _ in range(200):
        angle, radius = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 2)
        yield spec, Maratos, [radius * math.cos(angle), radius * math.sin(angle)]


def generated(problem, seeds):
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(300):
            yield problem(rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qp-solver", choices=["clarabel", "active-set"], default="active-set")
    args = parser.parse_args()
    families = [("stress", list(generated(sr.problem, range(1, 6)))),
                ("complementarity", list(generated(sm.problem, range(1, 3)))),
                ("maratos", list(maratos_problems()))]
    false_claims = []
    for name, problems in families:
        runs = {}
        for correction in (False, True):
            options = SqpOptions(max_iterations=300, qp_solver=args.qp_solver,
                                 second_order_correction=correction)
            statuses, records, tried, taken = collections.Counter(), [], 0, 0
            for k, (spec, model, start) in enumerate(problems):
                r = solve(spec, model(), start, method="sqp", method_options=options)
                statuses[r.status] += 1
                records.append((r.status, r.iterations, r.counts.functions + r.counts.derivatives))
                tried += r.report.second_order_corrections
                taken += r.report.second_order_steps
                if r.status == "converged":
                    check = cb.evaluate(cb.Case(name, spec, model, start, 0.0, math.inf),
                                        r.candidate, r.multipliers)
                    if not sr.first_order_ok(check):
                        false_claims.append((name, k, correction))
            runs[correction] = records
            print(f"{name:16s} correction {'on ' if correction else 'off'}: {dict(statuses)}, "
                  f"iterations {sum(x[1] for x in records)}, callbacks {sum(x[2] for x in records)}, "
                  f"corrections tried {tried}, taken {taken}")
        both = [(a, b) for a, b in zip(runs[False], runs[True]) if a[0] == b[0] == "converged"]
        print(f"{'':16s} both converged {len(both)}: iterations "
              f"{sum(a[1] for a, _ in both)} -> {sum(b[1] for _, b in both)}, callbacks "
              f"{sum(a[2] for a, _ in both)} -> {sum(b[2] for _, b in both)}")
    if false_claims:
        sys.exit(f"converged without passing the independent check: {false_claims}")


if __name__ == "__main__":
    main()
