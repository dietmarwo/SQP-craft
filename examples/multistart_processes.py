"""Many SQP starts of a Python model, one process per worker.

Python callbacks run on the thread that calls ``solve``, so the native batch
API cannot spread a Python model over threads. Distinct starts run in
separate processes instead: the ``ProblemSpec`` is picklable, every process
builds its own model, and each solve keeps one native thread
(``Options(threads=1)``), so ``workers`` processes use ``workers`` CPUs.

The problem is the six-hump camel function, smooth with six local minima,
inside a disk that cuts off two of them:

    minimize   (4 - 2.1 x^2 + x^4/3) x^2 + x y + (-4 + 4 y^2) y^2
    subject to x^2 + y^2 <= 3,   -3 <= x <= 3,  -2 <= y <= 2

The two global minima, f = -1.0316 at (+-0.0898, -+0.7126), lie inside the
disk. The best result is rechecked with ``kkt_residuals`` from a fresh model.

    python examples/multistart_processes.py --starts 32 --workers 4
"""

import argparse
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor

from sqp_craft import Evaluation, Options, ProblemSpec, SqpOptions, kkt_residuals, solve

LOWER, UPPER, RADIUS2 = [-3.0, -2.0], [3.0, 2.0], 3.0


class SixHumpCamel:
    def functions(self, x):
        a, b = x
        f = (4 - 2.1 * a**2 + a**4 / 3) * a**2 + a * b + (-4 + 4 * b**2) * b**2
        return f, [a**2 + b**2]

    def derivatives(self, x):
        a, b = x
        return [8 * a - 8.4 * a**3 + 2 * a**5 + b, a - 8 * b + 16 * b**3], [2 * a, 2 * b]


def make_spec():
    return ProblemSpec(lower=LOWER, upper=UPPER,
                       constraint_lower=[-float("inf")], constraint_upper=[RADIUS2],
                       jacobian_indptr=[0, 2], jacobian_indices=[0, 1])


def available_cpus():
    """The CPUs this process may run on: the affinity mask where the OS has
    one (Linux; it also reflects taskset and container CPU sets), else the
    CPU count. Python 3.13 also offers os.process_cpu_count()."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def solve_one(spec, start):
    """Runs in a worker process: a fresh model, one native thread."""
    return solve(spec, SixHumpCamel(), start, method="sqp",
                 method_options=SqpOptions(qp_solver="active-set", elastic_mode=True),
                 options=Options(threads=1))


def recheck(spec, x, multipliers, tolerance=1e-7):
    """First-order residuals at x from a fresh model, independent of the solver."""
    model = SixHumpCamel()
    f, c = model.functions(x)
    g, jac = model.derivatives(x)
    violation = max([0.0, c[0] - RADIUS2] + [max(lo - v, v - hi) for v, lo, hi in zip(x, LOWER, UPPER)])
    evaluation = Evaluation(objective=f, constraints=c, gradient=g, jacobian_values=jac,
                            primal_violation=violation, feasible_candidate=violation <= tolerance)
    return f, kkt_residuals(spec, x, evaluation, multipliers)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--starts", type=int, default=32, help="number of distinct starts (default 32)")
    parser.add_argument("--workers", type=int, help="worker processes (default: available CPUs, at most --starts)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    starts = [[rng.uniform(lo, hi) for lo, hi in zip(LOWER, UPPER)] for _ in range(args.starts)]
    workers = args.workers or min(available_cpus(), len(starts))
    # One thread per process also for libraries a model may use (NumPy's BLAS).
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    spec = make_spec()
    # "spawn" starts clean interpreters on every platform; the spec is pickled
    # to each process with every task.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(solve_one, [spec] * len(starts), starts))

    converged = [r for r in results if r.status == "converged"]
    print(f"{len(starts)} starts on {workers} processes: {len(converged)} converged")
    minima = sorted({(round(r.evaluation.objective, 4), round(r.candidate[0], 3), round(r.candidate[1], 3))
                     for r in converged})
    for f, a, b in minima:
        print(f"  local solution f = {f:9.4f} at ({a:6.3f}, {b:6.3f})")
    assert converged, "no start converged"
    best = min(converged, key=lambda r: r.evaluation.objective)
    f, k = recheck(spec, best.candidate, best.multipliers)
    print(f"best f = {f:.6f} at {best.candidate}; rechecked residuals: primal {k.primal:.1e}, "
          f"stationarity {k.stationarity:.1e}, complementarity {k.complementarity:.1e}, dual sign {k.dual_sign:.1e}")
    assert k.primal <= 1e-7 and k.stationarity <= 1e-6 and k.complementarity <= 1e-6 and k.dual_sign <= 1e-8
    # A first-order point under tolerance: the best of these starts, not a
    # certified global optimum.
    if len(starts) >= 16:
        assert f < -1.03, f


if __name__ == "__main__":
    main()
