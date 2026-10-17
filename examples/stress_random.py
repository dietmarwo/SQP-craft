"""Random feasible problems: sqp-craft with and without restoration, and SciPy SLSQP.

    pip install '.[bench]'
    python examples/stress_random.py [--seed S] [--count N] [--reuse-workspace]
        [--qp-solver clarabel|active-set] [--table]

Each problem minimizes ||x - target||^2 on [-2, 2]^n (n = 2..7) subject to
1-3 equality rows and 0-3 upper-bounded rows of the form
``a . x + s ||x - y||^2``, built around a hidden point x* so every problem is
feasible. Starts are uniform in the box, so many linearizations are
inconsistent at the start. Results are judged by the independent evaluator of
compare_baseline.py (first-order residuals only: random problems have no
reference objective). The script fails if sqp-craft ever reports
``converged`` at a point the evaluator rejects.
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compare_baseline as cb  # noqa: E402

from sqp_craft import SqpOptions, solve  # noqa: E402


def problem(rng):
    n = int(rng.integers(2, 8))
    m_eq = int(rng.integers(1, min(n, 4)))
    m_in = int(rng.integers(0, 4))
    hidden = rng.uniform(-1.5, 1.5, n)
    target = rng.uniform(-3, 3, n)
    rows = [(rng.normal(size=n), rng.uniform(0.1, 1.5), rng.uniform(-2, 2, n))
            for _ in range(m_eq + m_in)]
    values = [float(a @ hidden + s * np.sum((hidden - y) ** 2)) for a, s, y in rows]
    lower = values[:m_eq] + [-math.inf] * m_in
    upper = values[:m_eq] + [v + rng.uniform(0, 1) for v in values[m_eq:]]

    class Model:
        def functions(self, x):
            x = np.asarray(x)
            c = [float(a @ x + s * np.sum((x - y) ** 2)) for a, s, y in rows]
            return float(np.sum((x - target) ** 2)), c

        def derivatives(self, x):
            x = np.asarray(x)
            jac = np.concatenate([a + 2 * s * (x - y) for a, s, y in rows])
            return list(2 * (x - target)), list(jac)

    spec = cb.dense([-2.0] * n, [2.0] * n, lower, upper)
    return spec, Model, list(rng.uniform(-2, 2, n))


def first_order_ok(check) -> bool:
    return bool(check and check["primal"] <= cb.PRIMAL and check["stationarity"] <= cb.STATIONARITY
                and check["complementarity"] <= cb.COMPLEMENTARITY and check["dual_sign"] <= cb.SIGN)


def run(seed: int, count: int, reuse_workspace: bool = False, qp_solver: str = "clarabel"):
    rng = np.random.default_rng(seed)
    tally = collections.Counter()
    false_claims = []
    for k in range(count):
        spec, model, start = problem(rng)
        case = cb.Case("random", spec, model, start, 0.0, math.inf)
        common = dict(max_iterations=300, qp_reuse_workspace=reuse_workspace, qp_solver=qp_solver)
        for label, settings in (("sqp-craft", SqpOptions(**common)),
                                ("sqp-craft, restoration off",
                                 SqpOptions(max_restoration_steps=0, **common))):
            r = solve(spec, model(), start, method="sqp", method_options=settings)
            ok = first_order_ok(cb.evaluate(case, r.candidate, r.multipliers)
                                if r.candidate is not None else None)
            tally[(label, r.status, ok)] += 1
            if r.status == "converged" and not ok:
                false_claims.append((k, label))
        tally[("scipy-SLSQP", "accepted" if first_order_ok(cb.run_slsqp(case)["check"])
               else "not accepted", None)] += 1
    return tally, false_claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--reuse-workspace", action="store_true")
    parser.add_argument("--qp-solver", choices=["clarabel", "active-set"], default="clarabel")
    parser.add_argument("--table", action="store_true", help="print a Markdown table")
    args = parser.parse_args()
    tally, false_claims = run(args.seed, args.count, args.reuse_workspace, args.qp_solver)
    if args.table:
        print(f"Seed {args.seed}, {args.count} problems, QP solver {args.qp_solver}, "
              f"workspace reuse {'on' if args.reuse_workspace else 'off'}.\n")
        print("| Solver | Status | Independently accepted | Count |")
        print("| --- | --- | :---: | ---: |")
    for (solver, status, ok), n in sorted(tally.items(), key=lambda kv: (kv[0][0], -kv[1])):
        mark = "" if ok is None else ("yes" if ok else "no")
        print(f"| {solver} | {status} | {mark} | {n} |" if args.table else f"{solver:28s} {status:26s} {mark:4s} {n}")
    if false_claims:
        sys.exit(f"converged without passing the independent check: {false_claims}")


if __name__ == "__main__":
    main()
