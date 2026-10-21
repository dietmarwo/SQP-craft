"""Many SQP starts of a Python model, one process per worker.

Python callbacks run on the thread that calls ``solve``, so the native batch
API cannot spread a Python model over threads. Distinct starts run in
separate processes instead: the ``ProblemSpec`` is picklable, every process
builds its own model, and each solve keeps one native thread
(``Options(threads=1)``). The thread pools a model's libraries may use (NumPy's
BLAS, OpenMP) are set to ``--model-threads`` (default 1) in every process,
overwriting inherited values, so ``workers`` processes use ``workers *
model_threads`` CPUs. The default worker count keeps that within the CPUs this
process may use: its affinity mask and its cgroup CPU quota.

The problem is the six-hump camel function, smooth with six local minima,
inside a disk that cuts off two of them:

    minimize   (4 - 2.1 x^2 + x^4/3) x^2 + x y + (-4 + 4 y^2) y^2
    subject to x^2 + y^2 <= 3,   -3 <= x <= 3,  -2 <= y <= 2

The two global minima, f = -1.0316 at (+-0.0898, -+0.7126), lie inside the
disk. The best result is rechecked with ``kkt_residuals`` from a fresh model.

    python examples/multistart_processes.py --starts 32 --workers 4
"""

import argparse
import math
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def available_cpus(cgroup_file=Path("/proc/self/cgroup"), mount=Path("/sys/fs/cgroup")):
    """The CPUs this process may use: the affinity mask where the OS has one
    (Linux; taskset and container CPU sets), else the CPU count, lowered to
    the cgroup CPU quota (v2 ``cpu.max``, v1 ``cpu.cfs_quota_us``; the most
    restrictive one on the cgroup path), as optimization-run's node doctor
    counts them."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = cgroup_cpu_quota(cgroup_file, mount)
    return max(1, min(cpus, math.floor(quota)) if quota else cpus)


def cgroup_cpu_quota(cgroup_file=Path("/proc/self/cgroup"), mount=Path("/sys/fs/cgroup")):
    """The CPU quota of this process's cgroup in CPUs, or None."""
    try:
        lines = Path(cgroup_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    quotas = []
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and not controllers:
            bases, read = [Path(mount)], _quota_v2
        elif "cpu" in controllers.split(","):
            bases, read = [Path(mount) / controllers, Path(mount) / "cpu"], _quota_v1
        else:
            continue
        for base in bases:
            directory = base / path.lstrip("/")
            if not directory.exists():
                directory = base  # a cgroup namespace: read the mount root
            while True:
                quota = read(directory)
                if quota:
                    quotas.append(quota)
                if directory == base or directory.parent == directory:
                    break
                directory = directory.parent
    return min(quotas) if quotas else None


def _quota_v2(directory):
    try:
        quota, period = (directory / "cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def _quota_v1(directory):
    try:
        quota = int((directory / "cpu.cfs_quota_us").read_text())
        period = int((directory / "cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


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
    parser.add_argument("--workers", type=int,
                        help="worker processes (default: usable CPUs / --model-threads, at most --starts)")
    parser.add_argument("--model-threads", type=int, default=1,
                        help="thread-pool size of the model's libraries in each process (default 1)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    starts = [[rng.uniform(lo, hi) for lo, hi in zip(LOWER, UPPER)] for _ in range(args.starts)]
    usable = available_cpus()
    workers = args.workers or max(1, min(usable // args.model_threads, len(starts)))
    # The model libraries' pools, overwriting inherited values: the spawned
    # processes start fresh interpreters, so their libraries initialize with
    # these settings.
    for name in THREAD_VARIABLES:
        os.environ[name] = str(args.model_threads)
    needed = workers * args.model_threads
    print(f"topology: {workers} processes x {args.model_threads} model threads = {needed} CPUs; "
          f"{usable} usable" + (" (oversubscribed by explicit choice)" if needed > usable else ""))
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
