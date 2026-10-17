"""Compare cold/reused QP workspaces with unchanged independent acceptance.

python examples/compare_workspace.py --repeats 5 > workspace-results.json
Requires the project's bench extras. No timing threshold is asserted.
"""
from __future__ import annotations

import argparse
import json
import platform
from statistics import median

import compare_baseline as cb
import scipy
from sqp_craft import __version__


def run(repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    suite = list(cb.cases())
    acceptance = []
    for case in suite:
        for reuse in (False, True):
            r = cb.run_sqp_craft(case, reuse_workspace=reuse)
            r.update(case=case.name, start=[float(x) for x in case.start][:4], reuse=reuse)
            acceptance.append(r)
            if r["status"] != "converged" or not r["check"]["accepted"]:
                raise RuntimeError(f"comparison case failed: {r}")
    timings = []
    summary = []
    for case in (c for c in suite if c.name.startswith("orbit-")):
        for rep in range(repeats):
            # Acceptance sweep above warms both paths. Alternate the measured
            # order to avoid systematically favoring the second invocation.
            for reuse in ((False, True) if rep % 2 == 0 else (True, False)):
                r = cb.run_sqp_craft(case, reuse_workspace=reuse)
                r.update(case=case.name, repeat=rep, reuse=reuse)
                timings.append(r)
                if r["status"] != "converged" or not r["check"]["accepted"]:
                    raise RuntimeError(f"timed case failed: {r}")
        groups = {reuse: [r for r in timings if r["case"] == case.name and r["reuse"] == reuse]
                  for reuse in (False, True)}
        medians = {reuse: median(r["seconds"] for r in rows) for reuse, rows in groups.items()}
        summary.append(dict(case=case.name, repeats=repeats,
                            cold_median_seconds=medians[False], reused_median_seconds=medians[True],
                            speed_ratio=medians[False] / medians[True],
                            phase_medians={str(reuse): {k: median(r["phases"][k] for r in rows)
                                           for k in rows[0]["phases"]} for reuse, rows in groups.items()}))
    return dict(version=__version__, platform=platform.platform(), python=platform.python_version(),
                numpy=cb.np.__version__, scipy=scipy.__version__,
                thresholds=dict(primal=cb.PRIMAL, stationarity=cb.STATIONARITY,
                                complementarity=cb.COMPLEMENTARITY, dual_sign=cb.SIGN),
                acceptance=acceptance, summary=summary, timings=timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.repeats), indent=2, allow_nan=False))
