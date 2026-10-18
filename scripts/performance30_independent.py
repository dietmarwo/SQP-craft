"""Original-callback, independent NumPy checks on declared native fixture sizes.
Run in fresh processes under the baseline/final Python package respectively.
"""

import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import numpy as np
import orbit_transfer as orbit
import compare_baseline as cb
import stress_random as sr
from sqp_craft import ProblemSpec, SqpOptions, solve, __version__


def fixture(family, n):
    if family == "orbit":
        spec, model, start = orbit.build(n)
        return cb.Case(f"orbit-{n}", spec, lambda: model, start, 0.0, math.inf, 350)
    restoration = family == "restoration"
    m = n if restoration else n // 2
    target = np.ones(n) if restoration else 1.0 + np.arange(n) % 3 * 0.1
    weights = np.ones(n) if restoration else 1.0 + np.arange(n) % 10
    rows = np.ones(m) if restoration else target.reshape(-1, 2).sum(axis=1)

    class Model:
        def functions(self, x):
            x = np.asarray(x)
            return float(0.5 * np.sum(weights * (x - target) ** 2)), (
                x * x if restoration else x.reshape(-1, 2).sum(axis=1)
            )

        def derivatives(self, x):
            x = np.asarray(x)
            return weights * (x - target), 2 * x if restoration else np.ones(n)

    spec = ProblemSpec(
        lower=np.full(n, 0.05),
        upper=np.full(n, 2.0),
        constraint_lower=rows,
        constraint_upper=rows,
        jacobian_indptr=np.arange(m + 1) * (1 if restoration else 2),
        jacobian_indices=np.arange(n),
    )
    return cb.Case(f"{family}-{n}", spec, Model, np.full(n, 0.1), 0.0, 1e-9, 350)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--development", action="store_true")
    a = p.parse_args()
    cases = (
        [("orbit", 20), ("orbit", 40), ("separable", 128), ("restoration", 64)]
        if a.development
        else [
            ("orbit", 30),
            ("orbit", 48),
            ("separable", 96),
            ("separable", 192),
            ("restoration", 48),
            ("restoration", 96),
        ]
    )
    records = []
    false_claims = []
    for family, size in cases:
        case = fixture(family, size)
        for backend in ["active-set", "clarabel"]:
            r = solve(
                case.spec,
                case.make_model(),
                case.start,
                method="sqp",
                method_options=SqpOptions(
                    max_iterations=350,
                    max_callback_calls=3000,
                    qp_reuse_workspace=True,
                    qp_solver=backend,
                ),
            )
            check = (
                cb.evaluate(case, r.candidate, r.multipliers)
                if r.candidate is not None
                else None
            )
            ok = (
                check is not None
                and sr.first_order_ok(check)
                and check["certificate"] == "solver"
            )
            if r.status == "converged" and not ok:
                false_claims.append([family, size, backend])
            records.append(
                dict(
                    family=family,
                    size=size,
                    backend=backend,
                    status=r.status,
                    accepted=ok,
                    check=check,
                    iterations=r.iterations,
                    candidate=r.candidate,
                    functions=r.counts.functions,
                    derivatives=r.counts.derivatives,
                )
            )
            print(case.name, backend, r.status, ok, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            dict(version=__version__, records=records, false_claims=false_claims),
            indent=2,
        )
        + "\n"
    )
    if false_claims:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
