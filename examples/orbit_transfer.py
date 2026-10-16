"""Maximum-radius low-thrust orbit transfer (Bryson & Ho) by trapezoidal
collocation, solved with the native SQP.

A spacecraft with constant thrust T starts on a circular orbit of radius 1
(canonical units, mu = 1) and steers its thrust angle phi(t) to reach the
largest circular orbit at t_f = 3.32. States are radius r, radial velocity u
and tangential velocity v; the mass is a known function of time. Initial
states are fixed through equal variable bounds. Reference values from SciPy
SLSQP on the same discretization: r(t_f) = 1.52206 (N = 20), 1.52449 (N = 40).

    python examples/orbit_transfer.py [N]
"""

from __future__ import annotations

import sys

import numpy as np

from sqp_craft import ProblemSpec, SqpOptions, check_derivatives, solve

THRUST, MASS_FLOW, FINAL_TIME = 0.1405, 0.0749, 3.32


def build(n_intervals: int):
    """Return ``(spec, model, x0)`` for ``n_intervals`` collocation intervals."""
    nodes = n_intervals + 1
    h = FINAL_TIME / n_intervals
    mass = 1.0 - MASS_FLOW * h * np.arange(nodes)

    def col(block: int, node: int) -> int:
        return block * nodes + node

    # Defect row (state, i) couples all variables at nodes i and i + 1.
    rows = [
        sorted(col(b, node) for b in range(4) for node in (i, i + 1))
        for _state in range(3)
        for i in range(n_intervals)
    ]
    rows.append([col(1, n_intervals)])                       # u(tf) = 0
    rows.append([col(0, n_intervals), col(2, n_intervals)])  # v(tf) = 1/sqrt(r(tf))
    indptr = np.cumsum([0] + [len(r) for r in rows])
    indices = np.concatenate(rows)

    def unpack(x):
        x = np.asarray(x)
        return x[:nodes], x[nodes:2 * nodes], x[2 * nodes:3 * nodes], x[3 * nodes:]

    def dynamics(r, u, v, phi):
        s, c = np.sin(phi), np.cos(phi)
        f = np.array([u, v**2 / r - 1 / r**2 + THRUST * s / mass, -u * v / r + THRUST * c / mass])
        zero, one = np.zeros(nodes), np.ones(nodes)
        partials = np.array([  # [state][variable block][node]
            [zero, one, zero, zero],
            [-v**2 / r**2 + 2 / r**3, zero, 2 * v / r, THRUST * c / mass],
            [u * v / r**2, -v / r, -u / r, -THRUST * s / mass],
        ])
        return f, partials

    class OrbitTransfer:
        def functions(self, x):
            r, u, v, phi = unpack(x)
            f, _ = dynamics(r, u, v, phi)
            states = np.array([r, u, v])
            defects = states[:, 1:] - states[:, :-1] - 0.5 * h * (f[:, 1:] + f[:, :-1])
            terminal = [u[-1], v[-1] - 1 / np.sqrt(r[-1])]
            return -r[-1], np.concatenate([defects.ravel(), terminal])

        def derivatives(self, x):
            r, u, v, phi = unpack(x)
            _, d = dynamics(r, u, v, phi)
            gradient = np.zeros(4 * nodes)
            gradient[n_intervals] = -1.0
            values = []
            for state in range(3):
                for i in range(n_intervals):
                    for column in rows[state * n_intervals + i]:
                        block, node = divmod(column, nodes)
                        value = -0.5 * h * d[state][block][node]
                        if block == state:
                            value += 1.0 if node == i + 1 else -1.0
                        values.append(value)
            values += [1.0, 0.5 * r[-1] ** -1.5, 1.0]
            return gradient, values

    lower = np.repeat([0.5, -1.0, 0.1, -np.pi], nodes)
    upper = np.repeat([5.0, 1.0, 2.0, np.pi], nodes)
    for block, value in ((0, 1.0), (1, 0.0), (2, 1.0)):   # fixed initial state
        lower[col(block, 0)] = upper[col(block, 0)] = value
    m = len(rows)
    spec = ProblemSpec(lower=lower, upper=upper, constraint_lower=np.zeros(m),
                       constraint_upper=np.zeros(m), jacobian_indptr=indptr, jacobian_indices=indices)
    s = np.arange(nodes) / n_intervals
    x0 = np.concatenate([1 + 0.5 * s, np.zeros(nodes), 1 - 0.2 * s,
                         np.where(np.arange(nodes) < n_intervals / 2, 0.5, -0.5)])
    return spec, OrbitTransfer(), x0


def main(n_intervals: int = 40) -> None:
    spec, model, x0 = build(n_intervals)
    check = check_derivatives(spec, model, x0)
    print(f"derivative check: passed={check.passed}, {check.checked} coordinates, "
          f"fixed={check.fixed}, {check.counts.functions} function calls")
    assert check.passed

    result = solve(spec, model, x0, method="sqp",
                   method_options=SqpOptions(max_iterations=300, record_history=True))
    radius = -result.evaluation.objective
    report = result.report
    print(f"{result.status}: r(tf) = {radius:.5f} after {result.iterations} iterations "
          f"({result.counts.functions} function calls, {report.rejected_trials} rejected trials)")
    print(f"stationarity {result.stationarity:.1e}, primal violation "
          f"{result.evaluation.primal_violation:.1e}")
    print(f"time {report.total_seconds:.2f} s: QP {report.qp_seconds:.2f} s, "
          f"callbacks {report.callback_seconds:.2f} s, BFGS {report.bfgs_seconds:.2f} s")
    full = sum(1 for h in report.history if h.step_length == 1.0)
    print(f"full steps: {full} of {result.iterations}")
    assert result.status == "converged"


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
