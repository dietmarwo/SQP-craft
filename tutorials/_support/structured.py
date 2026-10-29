"""Canonical SQP1--SQP6 direct-API teaching models and experiments."""

from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp
from numerics import Case, case_from_model, plain, require, write


def calibration(preset):
    root = Path(__file__).resolve().parents[2]
    data = np.loadtxt(root / "tutorials/job-api/decay.csv", delimiter=",", skiprows=1)
    train = data[::2]
    hold = data[1::2]
    t, y = train.T
    T = 1.0
    Y = y[t == T][0]

    def fun(x):
        a, b, c = x
        r = a * np.exp(-b * t) + c - y
        return 0.5 * r @ r, np.array([a * np.exp(-b * T) + c])

    def derivative(x):
        a, b, c = x
        E = np.exp(-b * t)
        r = a * E + c - y
        J = np.array([E, -a * t * E, np.ones_like(t)]).T
        return J.T @ r, np.array([[np.exp(-b * T), -a * T * np.exp(-b * T), 1.0]])

    starts = [[5, 2.5, 0], [1, 0.2, 2], [9, 4.5, -4], [3, 1, 1]]
    case = Case([0, 0, -5], [10, 5, 5], [Y], [Y], fun, derivative, starts[0])
    records = []
    for backend in ["active-set", "clarabel"]:
        for start in starts:
            r, rec = case.run(backend, start=start)
            x = np.array(r.candidate)
            rec["holdout_half_sse"] = float(
                0.5
                * np.sum((x[0] * np.exp(-x[1] * hold[:, 0]) + x[2] - hold[:, 1]) ** 2)
            )
            records.append(rec)

    def reduced(b):
        E = np.exp(-b * T)
        z = np.exp(-b * t) - E
        d = y - Y
        lo = max(0, (Y - 5) / E)
        hi = min(10, (Y + 5) / E)
        if lo > hi:
            return np.inf
        a = np.clip(z @ d / (z @ z), lo, hi) if z @ z > 1e-25 else lo
        return float(0.5 * np.sum((a * z - d) ** 2))

    grid = np.linspace(0, 5, 1001)
    profile = np.array([reduced(b) for b in grid])
    k = int(np.argmin(profile))
    ref = minimize_scalar(
        reduced,
        bounds=(grid[max(0, k - 1)], grid[min(1000, k + 1)]),
        method="bounded",
        options={"xatol": 1e-13},
    )
    successful = [z for z in records if z["claim"] == "first_order"]
    require(successful)
    require(min(z["independent"]["objective"] for z in successful) <= ref.fun + 1e-7)
    old_y, old_Y = y, Y
    y = 3 * np.exp(-0.7 * t) + 0.5
    Y = 3 * np.exp(-0.7 * T) + 0.5
    noiseless = []
    zero_case = Case([0, 0, -5], [10, 5, 5], [Y], [Y], fun, derivative, starts[0])
    for backend in ["active-set", "clarabel"]:
        for initial in [starts[0], starts[1], starts[3]]:
            _, record = zero_case.run(backend, start=initial)
            noiseless.append(record)
        require(
            (
                any(
                    z["backend"] == backend
                    and z["claim"] == "first_order"
                    and z["independent"]["objective"] <= 1e-10
                    for z in noiseless
                )
            )
        )
    y, Y = old_y, old_Y
    return dict(
        training=train,
        holdout=hold,
        anchor=[T, Y],
        records=records,
        noiseless=noiseless,
        reduced_reference=dict(b=ref.x, loss=ref.fun),
        profile=dict(b=grid, loss=profile),
    )


def well(alpha=0.2, D=(1.0, 1.0), R=1.0):
    D = np.asarray(D)

    def fun(u):
        x, y = D * u
        return (x * x - 1) ** 2 + (y - x) ** 2 + alpha * x, np.array([R * (y - x)])

    def derivative(u):
        x, y = D * u
        return D * np.array(
            [4 * x * (x * x - 1) - 2 * (y - x) + alpha, 2 * (y - x)]
        ), np.array([R * np.array([-1, 1]) * D])

    return Case(
        -2 / D,
        2 / D,
        [0],
        [0],
        fun,
        derivative,
        np.array([-0.5, -0.5]) / D,
        linear_rows=[0],
    )


def diagnostics(preset):
    records = []
    roots = [
        brentq(lambda x: 4 * x * (x * x - 1) + 0.2, a, b)
        for a, b in [(-1.1, -1), (0, 0.1), (0.9, 1)]
    ]
    starts = [[-1.5, -1.5], [-0.5, -0.5], [0.5, 0.5], [1.5, 1.5]]
    for D, R in [((1, 1), 1), ((2**-8, 2**8), 64)]:
        for backend in ["active-set", "clarabel"]:
            for scaling in (
                ["none", "linear", "jacobian"]
                if preset == "study"
                else ["none", "jacobian"]
            ):
                for start in starts:
                    case = well(D=D, R=R)
                    r, rec = case.run(
                        backend, scaling, start=np.array(start) / D, check=True
                    )
                    physical = np.array(r.candidate) * D
                    rec.update(D=D, R=R, start=start, physical=physical)
                    if r.multipliers is not None:
                        from types import SimpleNamespace

                        mp = SimpleNamespace(
                            constraints=np.asarray(r.multipliers.constraints) * R,
                            variables=np.asarray(r.multipliers.variables) / D,
                        )
                        rec["physical_check"] = well().check(physical, mp)
                    # The native claim holds in the submitted coordinates; the
                    # tutorial's conclusion needs the physical-unit check too.
                    rec["physical_claim"] = (
                        "first_order"
                        if rec["claim"] == "first_order"
                        and rec.get("physical_check", {}).get("first_order")
                        else "feasible"
                        if rec.get("physical_check", {}).get("feasible")
                        else "none"
                    )
                    records.append(rec)
    zero = well(alpha=0)
    r, stationary = zero.run(start=[0, 0])
    # Along y = x the reduced objective is (x^2 - 1)^2, with curvature
    # 12 x^2 - 4 at the stationary point x = 0.
    x0 = float(np.array(r.candidate)[0])
    stationary["reduced_curvature"] = float(12 * x0**2 - 4)
    require(stationary["reduced_curvature"] < 0, stationary["reduced_curvature"])
    require(abs(stationary["independent"]["objective"] - 1) < 1e-10)
    simplex = []

    def sf(x):
        return np.dot(x, x), [sum(x)]

    def sd(x):
        return 2 * x, np.ones((1, 4))

    calibration = Case(np.zeros(4), np.ones(4), [1], [1], sf, sd, [1, 0, 0, 0])
    for backend in ["active-set", "clarabel"]:
        for start in [[1, 0, 0, 0], [0, 0, 0, 0]]:
            rr, record = calibration.run(backend, start=start)
            require(
                rr.status == "converged" and abs(rr.evaluation.objective - 0.25) < 1e-7
            )
            simplex.append(record)
    return dict(
        roots=roots,
        root_values=[(z * z - 1) ** 2 + 0.2 * z for z in roots],
        records=records,
        claim_counts={
            representation: {
                "native_first_order": sum(
                    r["claim"] == "first_order" for r in records if r["R"] == R
                ),
                "physical_first_order": sum(
                    r["physical_claim"] == "first_order" for r in records if r["R"] == R
                ),
                "solves": sum(r["R"] == R for r in records),
            }
            for representation, R in [("balanced", 1), ("ill_scaled", 64)]
        },
        simplex=simplex,
        stationary_maximum=stationary,
    )


def chain_reference(N, L=2.0, D=1.0):
    ell = L / N
    q = ell * (np.arange(N) - (N - 1) / 2)
    H = brentq(lambda h: np.sum(ell * h / np.sqrt(h * h + q * q)) - D, 1e-12, 100.0)
    dx = ell * H / np.sqrt(H * H + q * q)
    dy = ell * q / np.sqrt(H * H + q * q)
    return np.r_[0, np.cumsum(dx)], np.r_[0, np.cumsum(dy)]


def chain_case(N):
    ell = 2 / N
    n = N - 1

    def unpack(z):
        return np.r_[0, z[:n], 1], np.r_[0, z[n:], 0]

    def fun(z):
        x, y = unpack(z)
        return ell * np.sum(y[1:-1]), np.diff(x) ** 2 + np.diff(y) ** 2

    def derivative(z):
        x, y = unpack(z)
        J = np.zeros((N, 2 * n))
        g = np.r_[np.zeros(n), np.full(n, ell)]
        for k in range(N):
            for node, sign in [(k, -1), (k + 1, 1)]:
                if 0 < node < N:
                    J[k, node - 1] = sign * 2 * (x[k + 1] - x[k])
                    J[k, n + node - 1] = sign * 2 * (y[k + 1] - y[k])
        return g, J

    s = np.arange(1, N) / N
    start = np.r_[s, -3.2 * s * (1 - s)]
    pattern = [[j for j in range(2 * n) if (j % n) + 1 in (k, k + 1)] for k in range(N)]
    return Case(
        np.r_[np.full(n, -1.0), np.full(n, -1.5)],
        np.r_[np.full(n, 2.0), np.full(n, 1.5)],
        np.full(N, ell * ell),
        np.full(N, ell * ell),
        fun,
        derivative,
        start,
        pattern=pattern,
    )


def hanging(preset):
    a = brentq(lambda a: 2 * a * np.sinh(0.5 / a) - 2, 0.05, 1)
    s = 0.5 / a
    continuum = a * a * (s - np.sinh(s) * np.cosh(s))
    records = []
    for N in [16, 32, 64, 128] if preset == "study" else [16, 32]:
        case = chain_case(N)
        r, rec = case.run(iterations=500, seconds=120)
        xr, yr = chain_reference(N)
        reference = 2 / N * np.sum(yr[1:-1])
        rec.update(
            links=N,
            reference=reference,
            continuum=continuum,
            discretization_error=reference - continuum,
        )
        if r.status == "converged":
            require(abs(r.evaluation.objective - reference) < 1e-6)
        rec["x"] = np.r_[0, np.asarray(r.candidate)[: N - 1], 1]
        rec["y"] = np.r_[0, np.asarray(r.candidate)[N - 1 :], 0]
        records.append(rec)
    return dict(
        catenary_a=a, continuum_potential=continuum, records=records, maximum_links=129
    )


def brach_case(N):
    K = N + 1
    h = 1 / N
    n = 4 * K + 1
    g = 9.81
    phi = brentq(lambda q: (q - np.sin(q)) / (1 - np.cos(q)) - 1, 0.1, 3.1)
    radius = 1 / (1 - np.cos(phi))
    T = phi * np.sqrt(radius / g)
    q = np.linspace(0, phi, K)
    start = np.r_[
        radius * (q - np.sin(q)),
        radius * (1 - np.cos(q)),
        2 * np.sqrt(g * radius) * np.sin(q / 2),
        q / 2,
        T,
    ]

    def fields(z):
        x, y, v, theta = z[:-1].reshape(4, K)
        return np.array([v * np.sin(theta), v * np.cos(theta), g * np.cos(theta)])

    def fun(z):
        states = z[: 3 * K].reshape(3, K)
        F = fields(z)
        return z[-1], (
            states[:, 1:] - states[:, :-1] - 0.5 * h * z[-1] * (F[:, 1:] + F[:, :-1])
        ).ravel()

    def derivative(z):
        x, y, v, theta = z[:-1].reshape(4, K)
        F = fields(z)
        J = np.zeros((3 * N, n))
        grad = np.zeros(n)
        grad[-1] = 1
        for state in range(3):
            for k in range(N):
                row = state * N + k
                J[row, -1] = -0.5 * h * (F[state, k] + F[state, k + 1])
                for node in (k, k + 1):
                    J[row, state * K + node] += 1 if node == k + 1 else -1
                    dv = [np.sin(theta[node]), np.cos(theta[node]), 0][state]
                    dt = [
                        v[node] * np.cos(theta[node]),
                        -v[node] * np.sin(theta[node]),
                        -g * np.sin(theta[node]),
                    ][state]
                    J[row, 2 * K + node] -= 0.5 * h * z[-1] * dv
                    J[row, 3 * K + node] -= 0.5 * h * z[-1] * dt
        return grad, J

    lo = np.r_[np.zeros(3 * K), np.zeros(K), 0.2]
    hi = np.r_[np.full(K, 2), np.full(K, 2), np.full(K, 10), np.full(K, np.pi / 2), 2]
    for j in [0, K, 2 * K]:
        lo[j] = hi[j] = 0
    for j in [K - 1, 2 * K - 1]:
        lo[j] = hi[j] = 1
    pattern = [
        sorted(
            [
                state * K + k,
                state * K + k + 1,
                2 * K + k,
                2 * K + k + 1,
                3 * K + k,
                3 * K + k + 1,
                n - 1,
            ]
        )
        for state in range(3)
        for k in range(N)
    ]
    pattern = [sorted(set(r)) for r in pattern]
    return Case(
        lo,
        hi,
        np.zeros(3 * N),
        np.zeros(3 * N),
        fun,
        derivative,
        np.clip(start, lo, hi),
        pattern=pattern,
    ), T


ORBIT_REFERENCE = {20: 1.52206, 40: 1.52449}


def collocation(preset):
    records = []
    for N in [10, 20, 40] if preset == "study" else [10, 20]:
        case, Tref = brach_case(N)
        r, rec = case.run(seconds=120)
        z = np.array(r.candidate)
        K = N + 1
        T = z[-1]
        theta = z[3 * K : 4 * K]

        def rhs(t, s):
            th = np.interp(t, np.linspace(0, T, K), theta)
            return [s[2] * np.sin(th), s[2] * np.cos(th), 9.81 * np.cos(th)]

        replay = solve_ivp(
            rhs, [0, T], [0, 0, 0], method="DOP853", rtol=1e-11, atol=1e-12
        )
        rec.update(
            intervals=N,
            cycloid_time=Tref,
            time_error=float(T - Tref),
            replay_endpoint=replay.y[:, -1],
            replay_error=float(np.max(np.abs(replay.y[:2, -1] - 1))),
        )
        require(r.status == "converged", (N, r.status))
        records.append(rec)
    # A finer mesh must move both the descent time and the replayed endpoint
    # toward the exact answer.
    for coarse, fine in zip(records, records[1:]):
        require(
            abs(fine["time_error"]) < abs(coarse["time_error"]),
            "the descent time does not approach the cycloid",
        )
        require(
            fine["replay_error"] < coarse["replay_error"],
            "the replayed endpoint does not improve with the mesh",
        )
    root = Path(__file__).resolve().parents[2]
    sp = importlib.util.spec_from_file_location(
        "orbit_tutorial", root / "examples/orbit_transfer.py"
    )
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    orbit = []
    from sqp_craft import solve, SqpOptions, check_derivatives

    for N in [20, 40, 60] if preset == "study" else [20]:
        spec, model, x = mod.build(N)
        check = check_derivatives(spec, model, x)
        require(check.passed)
        r = solve(
            spec,
            model,
            x,
            method="sqp",
            method_options=SqpOptions(
                qp_solver="active-set",
                max_iterations=500,
                max_callback_calls=20000,
                time_limit_seconds=120,
            ),
        )
        require(r.status == "converged", (N, r.status))
        audit = case_from_model(spec, model, x).check(r.candidate, r.multipliers)
        require(audit.get("first_order"), audit)
        z = np.array(r.candidate)
        K = N + 1
        # Interpolate the unwrapped angle: the control crosses +-pi, and plain
        # interpolation between 2.7 and -1.8 rad would swing the thrust the
        # wrong way round.
        phi = np.unwrap(z[3 * K :])
        times = np.linspace(0, mod.FINAL_TIME, K)

        def rhs(t, s):
            radius, u, v = s
            angle = np.interp(t, times, phi)
            mass = 1 - mod.MASS_FLOW * t
            return [
                u,
                v * v / radius - 1 / radius**2 + mod.THRUST * np.sin(angle) / mass,
                -u * v / radius + mod.THRUST * np.cos(angle) / mass,
            ]

        replay = solve_ivp(
            rhs, [0, mod.FINAL_TIME], [1, 0, 1], method="DOP853", rtol=1e-11, atol=1e-12
        )
        final_radius = float(z[K - 1])
        reference = ORBIT_REFERENCE.get(N)
        if reference is not None:
            # SciPy SLSQP on the same discretization, from the example's
            # docstring, to the five decimals it states.
            require(
                abs(final_radius - reference) <= 5e-6,
                (N, final_radius, reference),
            )
        orbit.append(
            dict(
                intervals=N,
                native=plain(r),
                independent=audit,
                derivative_check=plain(check),
                final_radius=final_radius,
                slsqp_reference=reference,
                replay_endpoint=replay.y[:, -1],
                replay_radius_error=float(replay.y[0, -1] - final_radius),
                replay_terminal=[
                    replay.y[1, -1],
                    replay.y[2, -1] - 1 / np.sqrt(replay.y[0, -1]),
                ],
            )
        )
    for coarse, fine in zip(orbit, orbit[1:]):
        require(
            abs(fine["replay_radius_error"]) < abs(coarse["replay_radius_error"]),
            "the replayed radius does not approach the collocation radius",
        )
    return dict(brachistochrone=records, orbit=orbit)


def kinetics_states(p, times, step=0.02):
    A0, k1, k2 = p
    z = np.zeros(12)
    z[0] = A0
    z[3] = 1
    out = []
    now = 0.0

    def rhs(z):
        A, B, C = z[:3]
        S = z[3:].reshape(3, 3)
        Fy = np.array([[-k1, 0, 0], [k1, -k2, 0], [0, k2, 0]])
        Fp = np.array([[0, -A, 0], [0, A, -B], [0, 0, B]])
        return np.r_[[-k1 * A, k1 * A - k2 * B, k2 * B], (Fy @ S + Fp).ravel()]

    for t in times:
        count = max(1, math.ceil((t - now) / step))
        h = (t - now) / count
        for _ in range(count):
            k = rhs(z)
            l = rhs(z + h * k / 2)
            m = rhs(z + h * l / 2)
            q = rhs(z + h * m)
            z += h * (k + 2 * l + 2 * m + q) / 6
        now = t
        out.append(z.copy())
    return np.array(out)


def kinetics(preset):
    from numerics import complex_jac
    from sqp_craft import check_derivatives

    times = np.linspace(0, 8, 21)
    truth = np.array([1, 0.5, 1.5])
    obs = (np.exp(-0.5 * times) - np.exp(-1.5 * times)) * 0.5

    def model(t, observations, function_step=0.02, derivative_step=0.02):
        def fun(p):
            state = kinetics_states(p, t, function_step)
            r = state[:, 1] - observations
            return 0.5 * r @ r, []

        def derivative(p):
            state = kinetics_states(p, t, derivative_step)
            r = state[:, 1] - observations
            return state[:, 6:9].T @ r, np.zeros((0, 3))

        return Case(
            [0.05, 0.1, 0.1], [3, 3, 3], [], [], fun, derivative, [0.8, 0.4, 1.3]
        )

    case = model(times, obs)
    records = []
    for start in [[0.8, 0.4, 1.3], [0.4, 1.3, 0.4]]:
        r, rec = case.run(start=start)
        require(r.status == "converged")
        records.append(rec)
    swapped = np.array([1 / 3, 1.5, 0.5])
    sym = kinetics_states(truth, times)[:, 1] - kinetics_states(swapped, times)[:, 1]
    require(np.max(np.abs(sym)) < 1e-10)
    limit = kinetics_states([1, 1, 1], times)[:, 1]
    require(np.max(np.abs(limit - times * np.exp(-times))) < 1e-7)

    # Separate noisy fitting experiment. Odd sample times never enter the objective.
    seed, sigma = 20261029, 0.002
    all_times = np.linspace(0, 8, 41)
    clean = (np.exp(-0.5 * all_times) - np.exp(-1.5 * all_times)) * 0.5
    noisy = clean + np.random.default_rng(seed).normal(0, sigma, len(all_times))
    noisy_case = model(all_times[::2], noisy[::2])
    noisy_records = []
    for start in [[0.8, 0.4, 1.3], [0.4, 1.3, 0.4]]:
        r, rec = noisy_case.run(start=start)
        require(r.status == "converged")
        prediction = kinetics_states(r.candidate, all_times[1::2])[:, 1]
        rec["holdout_rmse"] = np.sqrt(np.mean((prediction - noisy[1::2]) ** 2))
        rec["holdout_clean_rmse"] = np.sqrt(np.mean((prediction - clean[1::2]) ** 2))
        noisy_records.append(rec)
    require(
        (
            noisy_records[0]["native"]["candidate"][1]
            < noisy_records[0]["native"]["candidate"][2]
        )
    )
    require(
        (
            noisy_records[1]["native"]["candidate"][1]
            > noisy_records[1]["native"]["candidate"][2]
        )
    )

    # Coarse functions and matching coarse sensitivities are consistent, but less
    # accurate physically. Mixing fine functions with coarse sensitivities fails.
    coarse = model(times, obs, 0.8, 0.8)
    inconsistent = model(times, obs, 0.02, 0.8)
    coarse_check = check_derivatives(coarse.spec, coarse, coarse.start)
    bad_check = check_derivatives(inconsistent.spec, inconsistent, inconsistent.start)
    require(coarse_check.passed and not bad_check.passed)
    fine_states = kinetics_states(truth, times)
    coarse_states = kinetics_states(truth, times, 0.8)
    closed_jac = complex_jac(
        lambda p: (
            p[0]
            * p[1]
            / (p[2] - p[1])
            * (np.exp(-p[1] * times) - np.exp(-p[2] * times))
        ),
        truth,
    )
    sensitivity_error = np.max(np.abs(fine_states[:, 6:9] - closed_jac))
    require(sensitivity_error < 1e-7)
    return dict(
        times=times,
        observations=obs,
        truth=truth,
        swapped=swapped,
        symmetry_error=np.max(np.abs(sym)),
        records=records,
        noisy_fit=dict(
            seed=seed,
            sigma=sigma,
            training_times=all_times[::2],
            training_observations=noisy[::2],
            holdout_times=all_times[1::2],
            holdout_observations=noisy[1::2],
            records=noisy_records,
        ),
        integration_audit=dict(
            fine_step=0.02,
            coarse_step=0.8,
            fine_closed_form_error=np.max(np.abs(fine_states[:, 1] - obs)),
            coarse_closed_form_error=np.max(np.abs(coarse_states[:, 1] - obs)),
            sensitivity_closed_form_error=sensitivity_error,
            coarse_consistency=plain(coarse_check),
            deliberately_inconsistent=plain(bad_check),
        ),
    )


CHAPTERS = {
    "SQP1": calibration,
    "SQP2": diagnostics,
    "SQP3": hanging,
    "SQP4": collocation,
    "SQP6": kinetics,
}


def main(chapter=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chapter", choices=list(CHAPTERS) + ["SQP5", "X2"], default=chapter
    )
    p.add_argument("--preset", choices=["smoke", "study"], default="smoke")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError("retain old results; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        if args.chapter in ["SQP5", "X2"]:
            from applications import rendezvous, power

            run = rendezvous if args.chapter == "SQP5" else power
        else:
            run = CHAPTERS[args.chapter]
        result = run(args.preset)
        write(
            args.output / "summary.json",
            dict(
                chapter=args.chapter,
                preset=args.preset,
                seconds=time.time() - start,
                result=result,
                implementation="native Rust SQP through Python callbacks",
            ),
        )
        print(
            json.dumps(
                dict(chapter=args.chapter, status="passed", output=str(args.output))
            )
        )
    except Exception as e:
        write(
            args.output / "failure.json",
            dict(
                chapter=args.chapter,
                error=repr(e),
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
