"""Random stress problems in three settings, judged by the independent evaluator.

    python3 scripts/validate_performance_stress.py out.json [--seed S] [--count N]

The same random family as examples/stress_random.py. Settings: cold QP
workspace, reused workspace, and reused workspace with restoration off.
Fails on any exception or on a `converged` result the evaluator rejects;
the JSON holds every per-problem result for pairing builds.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import compare_baseline as cb  # noqa: E402
import stress_random as sr  # noqa: E402

from sqp_craft import SqpOptions, __version__, solve  # noqa: E402

SETTINGS = [("cold", False, 10), ("reuse", True, 10), ("restoration_off", True, 0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = sr.np.random.default_rng(args.seed)
    records, errors, false_claims = [], [], []
    counts = collections.Counter()
    for case_id in range(args.count):
        spec, model, start = sr.problem(rng)
        case = cb.Case("random", spec, model, start, 0.0, math.inf)
        for mode, reuse, restoration in SETTINGS:
            options = SqpOptions(max_iterations=300, qp_reuse_workspace=reuse,
                                 max_restoration_steps=restoration)
            try:
                r = solve(spec, model(), start, method="sqp", method_options=options)
            except Exception as error:  # recorded, then fails the run below
                errors.append(dict(case=case_id, mode=mode, error=repr(error)))
                continue
            check = cb.evaluate(case, r.candidate, r.multipliers) if r.candidate is not None else None
            ok = sr.first_order_ok(check)
            if r.status == "converged" and (not ok or check["certificate"] != "solver"):
                false_claims.append([case_id, mode])
            counts[(mode, r.status, ok)] += 1
            records.append(dict(case=case_id, mode=mode, status=r.status, accepted=ok,
                                check=check, iterations=r.iterations))
    result = dict(version=__version__, seed=args.seed, cases=args.count,
                  solver_calls=len(records) + len(errors), false_claims=false_claims,
                  errors=errors,
                  counts=[dict(mode=m, status=s, accepted=a, count=n)
                          for (m, s, a), n in sorted(counts.items())],
                  records=records)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(__version__, args.seed, len(records), "errors", len(errors),
          "false claims", len(false_claims))
    if errors or false_claims:
        sys.exit(1)


if __name__ == "__main__":
    main()
