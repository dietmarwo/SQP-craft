"""Random problems with complementarity constraints, judged by the independent evaluator.

    python examples/stress_mpcc.py [--seed S] [--count N]
        [--qp-solver clarabel|active-set] [--elastic] [--table]

Each problem has k = 2..5 pairs (x_i, y_i) in [0, 10]² with x_i·y_i <= 0
(so x_i·y_i = 0) and one row Σx + Σy = s, and minimizes
½‖x − a‖² + ½‖y − b‖² for random a, b in [−1, 2]^k and s in [0.5, 3].
Every problem is feasible, but the Mangasarian–Fromovitz condition fails at
every feasible point, as for any complementarity constraint: the step QPs
are degenerate near every solution. Starts are uniform in
[0, 2]^2k. The script fails if a solve claims ``converged`` at a point the
evaluator of compare_baseline.py rejects. SciPy's SLSQP is not compared:
it returns no multipliers, and the evaluator's least-squares reconstruction
is not unique on these degenerate active sets.
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
import stress_random as sr  # noqa: E402

from sqp_craft import ProblemSpec, SqpOptions, solve  # noqa: E402


def problem(rng):
    k = int(rng.integers(2, 6))
    n = 2 * k
    a, b = rng.uniform(-1, 2, k), rng.uniform(-1, 2, k)
    total = rng.uniform(0.5, 3.0)

    class Model:
        def functions(self, z):
            z = np.asarray(z)
            x, y = z[:k], z[k:]
            return (float(0.5 * np.sum((x - a) ** 2) + 0.5 * np.sum((y - b) ** 2)),
                    list(x * y) + [float(z.sum())])

        def derivatives(self, z):
            z = np.asarray(z)
            x, y = z[:k], z[k:]
            jac = [v for i in range(k) for v in (y[i], x[i])] + [1.0] * n
            return list(np.concatenate([x - a, y - b])), jac

    indices = [j for i in range(k) for j in (i, k + i)] + list(range(n))
    spec = ProblemSpec(lower=[0.0] * n, upper=[10.0] * n,
                       constraint_lower=[-math.inf] * k + [total],
                       constraint_upper=[0.0] * k + [total],
                       jacobian_indptr=[2 * i for i in range(k + 1)] + [2 * k + n],
                       jacobian_indices=indices)
    return spec, Model, list(rng.uniform(0.0, 2.0, n))


def run(seed: int, count: int, qp_solver: str = "clarabel", elastic: bool = False):
    rng = np.random.default_rng(seed)
    tally = collections.Counter()
    false_claims = []
    options = SqpOptions(max_iterations=300, qp_solver=qp_solver, elastic_mode=elastic)
    for k in range(count):
        spec, model, start = problem(rng)
        case = cb.Case("mpcc", spec, model, start, 0.0, math.inf)
        r = solve(spec, model(), start, method="sqp", method_options=options)
        ok = sr.first_order_ok(cb.evaluate(case, r.candidate, r.multipliers)
                               if r.candidate is not None else None)
        tally[(r.status, ok)] += 1
        if r.status == "converged" and not ok:
            false_claims.append(k)
    return tally, false_claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--qp-solver", choices=["clarabel", "active-set"], default="clarabel")
    parser.add_argument("--elastic", action="store_true", help="elastic mode (active-set only)")
    parser.add_argument("--table", action="store_true", help="print a Markdown table")
    args = parser.parse_args()
    tally, false_claims = run(args.seed, args.count, args.qp_solver, args.elastic)
    label = args.qp_solver + (", elastic mode" if args.elastic else "")
    if args.table:
        print(f"Complementarity problems, seed {args.seed}, {args.count} problems, QP solver {label}.\n")
        print("| Status | Independently accepted | Count |")
        print("| --- | :---: | ---: |")
    for (status, ok), n in sorted(tally.items(), key=lambda kv: -kv[1]):
        mark = "yes" if ok else "no"
        print(f"| {status} | {mark} | {n} |" if args.table else f"{status:26s} {mark:4s} {n}")
    if false_claims:
        sys.exit(f"converged without passing the independent check: {false_claims}")


if __name__ == "__main__":
    main()
