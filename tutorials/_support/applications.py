"""Astrodynamics and power-flow applications, using explicit physical units."""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from numerics import Case, complex_jac, plain


def rendezvous(preset):
    import pykep_rust as pk

    N = 40 if preset == "study" else 20
    AU = 149597870700.0
    mu_si = 1.32712440018e20
    tu = np.sqrt(AU**3 / mu_si)
    vu = AU / tu
    epoch = 9800.0
    days = 300.0
    tof = days * 86400 / tu
    earth = np.asarray(pk.Planet.jpl_low_precision("earth").state(epoch))
    mars = np.asarray(pk.Planet.jpl_low_precision("mars").state(epoch + days))
    # The provider returns (position,velocity) in SI.
    dep = earth.ravel() / np.r_[np.full(3, AU), np.full(3, vu)]
    arr = mars.ravel() / np.r_[np.full(3, AU), np.full(3, vu)]
    n = 3 * N + 4
    thrust = 0.12
    ve = 30000 / vu

    def leg(z):
        state = dep.copy()
        state[3:] += z[-3:]
        return pk.SimsFlanaganLeg(
            state.tolist(),
            1.0,
            z[: 3 * N].reshape(N, 3).tolist(),
            arr.tolist(),
            z[3 * N],
            tof,
            thrust,
            ve,
            1.0,
            0.5,
        )

    def fun(z):
        l = leg(z)
        return -z[3 * N], np.r_[
            l.mismatch_constraints(), np.sum(z[: 3 * N].reshape(N, 3) ** 2, axis=1)
        ]

    def deriv(z):
        l = leg(z)
        jd, ja, jc = map(np.asarray, l.mismatch_jacobian())
        J = np.zeros((7 + N, n))
        J[:7, : 3 * N] = jc[:, : 3 * N]
        J[:7, 3 * N] = ja[:, 6]
        J[:7, -3:] = jd[:, 3:6]
        for i in range(N):
            J[7 + i, 3 * i : 3 * i + 3] = 2 * z[3 * i : 3 * i + 3]
        g = np.zeros(n)
        g[3 * N] = -1
        return g, J

    lambert = pk.LambertProblem(
        dep[:3], arr[:3], tof, 1.0, maximum_revolutions=0
    ).solutions[0]
    excess = np.asarray(lambert.departure_velocity) - dep[3:]
    assert np.all(np.abs(excess) < 0.25)
    arrival_burn = arr[3:] - np.asarray(lambert.arrival_velocity)
    seed = np.full((N, 3), 1e-4)
    seed[-N // 4 :] = arrival_burn / (thrust * tof / 4)
    start = np.r_[seed.ravel(), np.exp(-np.linalg.norm(arrival_burn) / ve), excess]
    case = Case(
        np.r_[np.full(3 * N, -1.0), 0.2, np.full(3, -0.25)],
        np.r_[np.ones(3 * N), 1.0, np.full(3, 0.25)],
        np.r_[np.zeros(7), np.full(N, -np.inf)],
        np.r_[np.zeros(7), np.ones(N)],
        fun,
        deriv,
        start,
    )
    r, rec = case.run(scaling="jacobian", iterations=500, seconds=180)
    exact_record = rec
    if r.status != "converged":
        # Separate declared model: exclude nonsmooth zero-throttle coasts.
        floor = 0.02
        smooth_start = np.asarray(r.candidate).copy()
        u = smooth_start[: 3 * N].reshape(N, 3)
        norms = np.linalg.norm(u, axis=1)
        for i in range(N):
            if norms[i] < floor:
                u[i] = (
                    u[i] / max(norms[i], 1e-30) * floor * 1.1
                    if norms[i] > 1e-15
                    else [floor * 1.1, 0, 0]
                )
        smooth = Case(
            case.lower,
            case.upper,
            np.r_[np.zeros(7), np.full(N, floor**2)],
            np.r_[np.zeros(7), np.ones(N)],
            fun,
            deriv,
            smooth_start,
        )
        r, rec = smooth.run(scaling="jacobian", iterations=1500, seconds=180)
        rec["minimum_throttle_constraint"] = floor
    z = np.array(r.candidate)
    controls = z[: 3 * N].reshape(N, 3)
    duration = tof / N
    cut = N // 2
    assert r.status == "converged", f"rendezvous did not converge: {r.status}"

    def propagate(state, dt):
        def rhs(t, s):
            return np.r_[s[3:6], -s[:3] / np.linalg.norm(s[:3]) ** 3]

        integration = solve_ivp(
            rhs, [0, dt], state[:6], method="DOP853", rtol=3e-12, atol=3e-13
        )
        assert integration.success
        state[:6] = integration.y[:, -1]

    def half(state, indices, direction):
        state = state.copy()
        for i in indices:
            propagate(state, direction * duration / 2)
            dv = thrust * duration * controls[i] / state[6]
            state[3:6] += direction * dv
            state[6] *= np.exp(-direction * np.linalg.norm(dv) / ve)
            propagate(state, direction * duration / 2)
        return state

    forward = half(np.r_[dep[:3], dep[3:] + z[-3:], 1.0], range(cut), 1)
    backward = half(np.r_[arr, z[3 * N]], range(N - 1, cut - 1, -1), -1)
    replay = forward - backward
    native = np.array(leg(z).mismatch_constraints())
    assert np.max(np.abs(replay - native)) < 1e-7
    if r.status == "converged":
        assert np.max(np.abs(replay)) < 2e-7
    return dict(
        result=rec,
        original_coast_model=exact_record,
        departure_mjd2000=epoch,
        flight_days=days,
        segments=N,
        seed="Lambert departure and distributed arrival correction",
        units=dict(length_m=AU, time_s=tu, mass_kg=1000),
        departure=dep,
        arrival=arr,
        throttles=controls,
        arrival_mass=z[3 * N],
        departure_excess=z[-3:],
        independent_mismatch=replay,
        mismatch_position_km=replay[:3] * AU / 1000,
        mismatch_velocity_m_s=replay[3:6] * vu,
        minimum_throttle_norm=float(np.min(np.linalg.norm(controls, axis=1))),
        nonsmooth_coast_warning=bool(np.any(np.linalg.norm(controls, axis=1) < 1e-6)),
        propagator="pykep Lagrangian Kepler drifts and analytic STM Jacobians; independent scipy DOP853 midpoint-impulse replay",
    )


def network():
    edges = [
        (0, 1, 0.01, 0.10),
        (1, 2, 0.01, 0.10),
        (2, 3, 0.01, 0.10),
        (3, 4, 0.01, 0.10),
        (4, 0, 0.01, 0.10),
        (1, 3, 0.015, 0.15),
    ]
    Y = np.zeros((5, 5), complex)
    for i, j, r, x in edges:
        y = 1 / complex(r, x)
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y
    return (
        edges,
        Y,
        np.array([0, 0.8, 0.5, 0.6, 0.3]),
        np.array([0, 0.25, 0.15, 0.2, 0.1]),
    )


def power(preset):
    from lp_ip_craft import Model, NativeOptions, Options, solve

    edges, Y, Pd, Qd = network()
    B = np.zeros((5, 5))
    gen = [0, 3]
    cost = np.array([1.0, 1.5])
    for i, j, r, x in edges:
        B[i, i] += 1 / x
        B[j, j] += 1 / x
        B[i, j] -= 1 / x
        B[j, i] -= 1 / x
    Aeq = np.zeros((5, 7))
    Aeq[:, :5] = B
    for k, i in enumerate(gen):
        Aeq[i, 5 + k] = -1
    line = np.zeros((len(edges), 7))
    for k, (i, j, r, x) in enumerate(edges):
        line[k, i] = 1 / x
        line[k, j] = -1 / x
    A = np.vstack([Aeq, line])
    ptr = np.arange(0, A.size + 1, 7)
    lp = Model(
        objective=[0] * 5 + cost.tolist(),
        lower=[0] + [-1] * 4 + [0, 0],
        upper=[0] + [1] * 4 + [3, 2],
        row_lower=np.r_[-Pd, np.full(len(edges), -1.5)],
        row_upper=np.r_[-Pd, np.full(len(edges), 1.5)],
        indptr=ptr,
        indices=np.tile(np.arange(7), len(A)),
        values=A.ravel(),
    )
    dc = solve(
        lp,
        backend="native",
        options=Options(threads=1),
        backend_options=NativeOptions(component_threads=1),
    )
    assert dc.status == "optimal"
    dcx = np.array(dc.candidate)

    def equations(z):
        V = z[:5]
        theta = z[5:10]
        Pg = z[10:12]
        Qg = z[12:14]
        delta = theta[:, None] - theta
        P = V * np.sum(
            V[None, :] * (Y.real * np.cos(delta) + Y.imag * np.sin(delta)), axis=1
        )
        Q = V * np.sum(
            V[None, :] * (Y.real * np.sin(delta) - Y.imag * np.cos(delta)), axis=1
        )
        p = P + Pd
        q = Q + Qd
        for k, i in enumerate(gen):
            p[i] -= Pg[k]
            q[i] -= Qg[k]
        thermal = []
        for i, j, r, x in edges:
            y = 1 / complex(r, x)
            g, b = y.real, y.imag
            for i, j in [(i, j), (j, i)]:
                d = theta[i] - theta[j]
                pf = g * V[i] ** 2 - V[i] * V[j] * (g * np.cos(d) + b * np.sin(d))
                qf = -b * V[i] ** 2 - V[i] * V[j] * (g * np.sin(d) - b * np.cos(d))
                thermal.append(pf * pf + qf * qf)
        return np.r_[p, q, thermal]

    def fun(z):
        return cost @ z[10:12], equations(z)

    def deriv(z):
        return np.r_[np.zeros(10), cost, np.zeros(2)], complex_jac(equations, z)

    lo = np.r_[np.full(5, 0.9), np.full(5, -1), [0, 0], [-2, -2]]
    hi = np.r_[np.full(5, 1.1), np.ones(5), [3, 2], [2, 2]]
    lo[5] = hi[5] = 0
    lo[0] = hi[0] = 1
    start = np.r_[np.ones(5), dcx[:5], dcx[5:], [0.4, 0.3]]
    case = Case(
        lo,
        hi,
        np.r_[np.zeros(10), np.full(12, -np.inf)],
        np.r_[np.zeros(10), np.full(12, 2.25)],
        fun,
        deriv,
        start,
    )
    r, rec = case.run(scaling="jacobian")
    assert r.status == "converged"
    z = np.array(r.candidate)
    voltage = z[:5] * np.exp(1j * z[5:10])
    S = voltage * np.conj(Y @ voltage)
    targetP = -Pd.copy()
    targetQ = -Qd.copy()
    for k, i in enumerate(gen):
        targetP[i] += z[10 + k]
        targetQ[i] += z[12 + k]
    assert (
        np.max(np.abs(S.real - targetP)) < 1e-7
        and np.max(np.abs(S.imag - targetQ)) < 1e-7
    )

    # Independent PQ Newton replay of the DC dispatch; slack absorbs losses.
    def residual(u):
        V = np.r_[1, u[4:]] * np.exp(1j * np.r_[0, u[:4]])
        S = V * np.conj(Y @ V)
        p = -Pd.copy()
        p[3] += dcx[6]
        return np.r_[S.real[1:] - p[1:], S.imag[1:] + Qd[1:]]

    replay = root(residual, np.r_[dcx[1:5], np.ones(4)])
    assert replay.success and np.max(np.abs(residual(replay.x))) < 1e-7
    return dict(
        edges=edges,
        loads_P=Pd,
        loads_Q=Qd,
        dc=plain(dc),
        ac=rec,
        dc_ac_replay=dict(
            angles=np.r_[0, replay.x[:4]],
            voltage=np.r_[1, replay.x[4:]],
            residual=residual(replay.x),
        ),
        ac_complex_balance_error=float(np.max(np.abs(S - (targetP + 1j * targetQ)))),
    )
