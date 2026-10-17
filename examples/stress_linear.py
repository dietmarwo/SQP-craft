"""Random problems whose model is undefined outside linear constraints.

    python examples/stress_linear.py [--seed S] [--count N]
        [--qp-solver clarabel|active-set] [--declare] [--table]

Each problem has event times t_0..t_K (K = 3..6) in [0, 20] with the phase
durations d_k = t_{k+1} − t_k. Three kinds of constraints apply:
- K linear rows d_k ≥ δ_k, with δ_k in [0.2, 0.6];
- one nonlinear equality Σ √d_k = S;
- one nonlinear inequality Σ log d_k ≥ L.

The objective is Σ (a_k d_k + b_k/d_k) + ½(t_0 − τ)². The model raises
UndefinedPointError when some d_k ≤ 0, as a trajectory model does when two
events swap order. S and L come from a hidden point with every d_k above
δ_k, so every problem is feasible.

Starts have durations uniform in [0.05, 2], so the model is defined there,
but most starts violate some linear row. With ``--declare`` the duration rows
are declared linear (``ProblemSpec(linear_rows=...)``): the solver then
projects the start onto them and keeps every iterate on them.

The script fails if a solve claims ``converged`` at a point the evaluator of
compare_baseline.py rejects.
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

from sqp_craft import ProblemSpec, SqpOptions, UndefinedPointError, solve  # noqa: E402


def problem(rng, declare: bool = False):
    k = int(rng.integers(3, 7))
    n = k + 1
    delta = rng.uniform(0.2, 0.6, k)
    a, b = rng.uniform(0.2, 2.0, k), rng.uniform(0.2, 2.0, k)
    tau = rng.uniform(0.0, 3.0)
    hidden = delta + rng.uniform(0.1, 1.5, k)
    total = float(np.sum(np.sqrt(hidden)))
    floor = float(np.sum(np.log(hidden)) - rng.uniform(0.0, 1.0))

    class Model:
        def durations(self, t):
            d = np.diff(np.asarray(t, dtype=float))
            if np.any(d <= 0.0):
                raise UndefinedPointError("event times out of order")
            return d

        def functions(self, t):
            d = self.durations(t)
            objective = float(np.sum(a * d + b / d) + 0.5 * (t[0] - tau) ** 2)
            return objective, list(d) + [float(np.sum(np.sqrt(d))), float(np.sum(np.log(d)))]

        def derivatives(self, t):
            d = self.durations(t)
            per_duration = a - b / d**2  # ∂f/∂d_k
            gradient = np.zeros(n)
            gradient[:-1] -= per_duration
            gradient[1:] += per_duration
            gradient[0] += t[0] - tau
            jac = [-1.0, 1.0] * k
            for weights in (0.5 / np.sqrt(d), 1.0 / d):  # ∂/∂d_k of the two rows
                row = np.zeros(n)
                row[:-1] -= weights
                row[1:] += weights
                jac += list(row)
            return list(gradient), jac

    rows = [[i, i + 1] for i in range(k)] + [list(range(n))] * 2
    spec = ProblemSpec(lower=[0.0] * n, upper=[20.0] * n,
                       constraint_lower=list(delta) + [total, floor],
                       constraint_upper=[math.inf] * k + [total, math.inf],
                       jacobian_indptr=list(np.cumsum([0] + [len(r) for r in rows])),
                       jacobian_indices=[j for r in rows for j in r],
                       linear_rows=list(range(k)) if declare else None)
    start = list(rng.uniform(0.0, 2.0) + np.concatenate([[0.0], np.cumsum(rng.uniform(0.05, 2.0, k))]))
    return spec, Model, start


def run(seed: int, count: int, qp_solver: str = "active-set", declare: bool = False):
    rng = np.random.default_rng(seed)
    tally, false_claims, undefined = collections.Counter(), [], 0
    options = SqpOptions(max_iterations=300, qp_solver=qp_solver)
    for k in range(count):
        spec, model, start = problem(rng, declare)
        case = cb.Case("linear", spec, model, start, 0.0, math.inf)
        r = solve(spec, model(), start, method="sqp", method_options=options)
        ok = sr.first_order_ok(cb.evaluate(case, r.candidate, r.multipliers)
                               if r.candidate is not None else None)
        tally[(r.status, ok)] += 1
        undefined += r.report.undefined_trials
        if r.status == "converged" and not ok:
            false_claims.append(k)
    return tally, false_claims, undefined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--qp-solver", choices=["clarabel", "active-set"], default="active-set")
    parser.add_argument("--declare", action="store_true", help="declare the duration rows linear")
    parser.add_argument("--table", action="store_true", help="print a Markdown table")
    args = parser.parse_args()
    tally, false_claims, undefined = run(args.seed, args.count, args.qp_solver, args.declare)
    if args.table:
        print(f"Linear-guard problems, seed {args.seed}, {args.count} problems, QP solver "
              f"{args.qp_solver}, linear rows {'declared' if args.declare else 'not declared'}; "
              f"{undefined} undefined trials.\n")
        print("| Status | Independently accepted | Count |")
        print("| --- | :---: | ---: |")
    for (status, ok), n in sorted(tally.items(), key=lambda kv: -kv[1]):
        mark = "yes" if ok else "no"
        print(f"| {status} | {mark} | {n} |" if args.table else f"{status:26s} {mark:4s} {n}")
    if not args.table:
        print(f"undefined trials {undefined}")
    if false_claims:
        sys.exit(f"converged without passing the independent check: {false_claims}")


if __name__ == "__main__":
    main()
