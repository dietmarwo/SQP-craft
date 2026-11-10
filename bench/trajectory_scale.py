"""Reproduce the frozen SQP5 single-leg and multiple-shooting study.

Run from the repository root with sqp-craft 0.18.0 or later and pykep-rust
0.1.6 or later (or both built from source) installed.
Each invocation runs one case so a shell deadline can bound each comparator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse.linalg import lsmr

from bench import solvers

AU = 149597870700.0
TU = np.sqrt(AU**3 / 1.32712440018e20)
VU = AU / TU
DAYS = 300.0
TOF = DAYS * 86400.0 / TU
THRUST = 0.12
VE = 30000.0 / VU
WIDTH = 4
PRIMAL_TOL = 1e-7
STATIONARITY_TOL = 1e-6
ITERATIONS = 1500
SECONDS = 30.0


class Transfer:
    def __init__(self, arcs, floor, single=False, seed="lambert", start_file=None):
        import pykep_rust as pk

        self.arcs, self.floor, self.single = arcs, floor, single
        self.segments = arcs * WIDTH
        units = [AU] * 3 + [VU] * 3
        self.dep = np.asarray(pk.Planet.jpl_low_precision("earth").state(9800)).ravel() / units
        self.arr = np.asarray(pk.Planet.jpl_low_precision("mars").state(9800 + DAYS)).ravel() / units
        lambert = pk.LambertProblem(
            self.dep[:3], self.arr[:3], TOF, 1.0, maximum_revolutions=0
        ).solutions[0]
        excess = np.asarray(lambert.departure_velocity) - self.dep[3:]
        arrival_burn = self.arr[3:] - np.asarray(lambert.arrival_velocity)
        start_controls = np.full((self.segments, 3), 1e-4)
        start_controls[-arcs:] = arrival_burn / (THRUST * TOF / 4)
        norms = np.linalg.norm(start_controls, axis=1)
        # The same initial controls are used at floor 0 and 0.02.
        start_controls[norms < 0.022] *= (0.022 / norms[norms < 0.022])[:, None]
        if seed == "offset":
            excess += np.array([0.004, -0.003, 0.002])
        mass = float(np.exp(-np.linalg.norm(arrival_burn) / VE))
        if start_file is not None:
            data = json.loads(start_file.read_text())
            controls = np.asarray(data["controls"])
            if controls.shape != start_controls.shape:
                raise ValueError("reference start has a different segment count")
            start_controls, mass, excess = controls, data["arrival_mass"], data["departure_excess"]
        if single:
            self._leg = pk.SimsFlanaganLeg
            self.n = 4 + 3 * self.segments
            self.elements = [list(range(7 + self.segments))]
            columns = [list(range(self.n)) for _ in range(7)]
            columns.extend(list(range(j, j + 3)) for j in range(3, self.n - 1, 3))
            self.indptr = np.cumsum([0, *map(len, columns)]).tolist()
            self.indices = [j for row in columns for j in row]
            self.x0 = np.r_[excess, start_controls.ravel(), mass]
            self.lower = np.r_[[-0.25] * 3, [-1.0] * (3 * self.segments), 0.2]
            self.upper = np.r_[[0.25] * 3, [1.0] * (3 * self.segments), 1.0]
            self.controls = [slice(3, self.n - 1)]
            self.nodes = []
            self.clower = np.r_[[0.0] * 7, [floor**2 if floor else -np.inf] * self.segments]
            self.cupper = np.r_[[0.0] * 7, [1.0] * self.segments]
        else:
            from pykep_rust.trajopt import SimsFlanaganChain

            self._chain = SimsFlanaganChain(self.dep, self.arr, 1.0, arcs, WIDTH, TOF, THRUST, VE, 1.0)
            self.n = self._chain.dimension
            self.elements = self._chain.elements
            self.indptr = self._chain.jacobian_indptr
            self.indices = self._chain.jacobian_indices
            self.x0 = np.asarray(self._chain.lift(start_controls, mass, excess))
            self.lower = np.r_[[-0.25] * 3, np.tile([-1.0] * (3 * WIDTH) + [-4.0] * 6 + [0.1], arcs - 1), [-1.0] * (3 * WIDTH), 0.2]
            self.upper = np.r_[[0.25] * 3, np.tile([1.0] * (3 * WIDTH) + [4.0] * 6 + [1.2], arcs - 1), [1.0] * (3 * WIDTH), 1.0]
            self.controls = [slice(3 + i * (3 * WIDTH + 7), 3 + i * (3 * WIDTH + 7) + 3 * WIDTH)
                             for i in range(arcs)]
            self.nodes = [slice(block.stop, block.stop + 7) for block in self.controls[:-1]]
            self.clower = np.tile([0.0] * 7 + [floor**2 if floor else -np.inf] * WIDTH, arcs)
            self.cupper = np.tile([0.0] * 7 + [1.0] * WIDTH, arcs)
        self.m = len(self.clower)
        self.nnz = len(self.indices)
        self._rows = np.repeat(np.arange(self.m), np.diff(self.indptr))
        self._cols = np.asarray(self.indices)
        self.g = np.zeros(self.n)
        self.g[-1] = -1.0
        self.reset_counts()
        self.track_counts = False

    def reset_counts(self):
        self.function_points, self.derivative_points = set(), set()

    @property
    def counts(self):
        return len(self.function_points), len(self.derivative_points)

    def spec(self, structured=False):
        from sqp_craft import ProblemSpec

        declarations = dict(elements=self.elements, objective_pattern=[]) if structured else {}
        return ProblemSpec(lower=self.lower, upper=self.upper, constraint_lower=self.clower,
                           constraint_upper=self.cupper, jacobian_indptr=self.indptr,
                           jacobian_indices=self.indices, **declarations)

    def single_leg(self, x):
        dep = self.dep.copy()
        dep[3:] += x[:3]
        return self._leg(dep, 1.0, x[3:-1].reshape(-1, 3), self.arr, x[-1],
                         TOF, THRUST, VE, 1.0, 0.5)

    def functions(self, x):
        x = np.asarray(x)
        if self.track_counts:
            self.function_points.add(tuple(x))
        if not self.single:
            f, rows = self._chain.evaluate(x)
            return f, np.asarray(rows)
        leg = self.single_leg(x)
        return -x[-1], np.r_[leg.mismatch_constraints(),
                             np.sum(x[3:-1].reshape(-1, 3)**2, axis=1)]

    def derivatives(self, x):
        x = np.asarray(x)
        if self.track_counts:
            self.derivative_points.add(tuple(x))
        if not self.single:
            return self.g, np.asarray(self._chain.jacobian_values(x))
        jd, ja, jc = map(np.asarray, self.single_leg(x).mismatch_jacobian())
        return self.g, np.r_[np.c_[jd[:, 3:6], jc[:, :-1], ja[:, 6]].ravel(), 2 * x[3:-1]]

    def objective(self, x):
        return -x[-1]

    def constraints(self, x):
        return self.functions(x)[1]

    def gradient(self, _x):
        return self.g

    def jacobian_values(self, x):
        return self.derivatives(x)[1]

    def dense_jacobian(self, values):
        return csr_matrix((values, self.indices, self.indptr), shape=(self.m, self.n)).toarray()

    def unpack_controls(self, x):
        return np.concatenate([x[block] for block in self.controls]).reshape(-1, 3)

    def replay(self, x):
        """Independent DOP853 Kepler drift and explicit midpoint impulses."""
        controls = self.unpack_controls(x)
        duration = TOF / self.segments
        dep = np.r_[self.dep[:3], self.dep[3:] + x[:3], 1.0]
        arr = np.r_[self.arr, x[-1]]

        def drift(state, dt):
            def rhs(_, s):
                return np.r_[s[3:6], -s[:3] / np.linalg.norm(s[:3])**3]

            result = solve_ivp(rhs, [0, dt], state[:6], method="DOP853", rtol=3e-12, atol=3e-13)
            if not result.success:
                raise ArithmeticError(result.message)
            state[:6] = result.y[:, -1]

        def half(state, indices, direction):
            state = state.copy()
            for j in indices:
                drift(state, direction * duration / 2)
                dv = THRUST * duration * controls[j] / state[6]
                state[3:6] += direction * dv
                state[6] *= np.exp(-direction * np.linalg.norm(dv) / VE)
                drift(state, direction * duration / 2)
            return state

        def defect(left, right, first, stop):
            cut = min(stop, max(first, self.segments // 2))
            return (half(left, range(first, cut), 1) -
                    half(right, range(stop - 1, cut - 1, -1), -1))

        global_defect = defect(dep, arr, 0, self.segments)
        endpoints = [dep, *(x[block] for block in self.nodes), arr]
        width = self.segments if self.single else WIDTH
        arc_defects = np.stack([defect(left, right, i * width, (i + 1) * width)
                                for i, (left, right) in enumerate(zip(endpoints, endpoints[1:]))])
        native = self.constraints(x)[self.clower == self.cupper].reshape(-1, 7)
        return dict(position_km=float(np.max(np.abs(global_defect[:3])) * AU / 1000),
                    velocity_m_s=float(np.max(np.abs(global_defect[3:6])) * VU),
                    mass_kg=float(abs(global_defect[6]) * 1000),
                    arc_position_km=float(np.max(np.abs(arc_defects[:, :3])) * AU / 1000),
                    arc_velocity_m_s=float(np.max(np.abs(arc_defects[:, 3:6])) * VU),
                    native_agreement=float(np.max(np.abs(arc_defects - native))))


def check(p, x, lam, z):
    """Reevaluate the point and first-order residuals with sparse operations."""
    _, c = p.functions(x)
    g, values = p.derivatives(x)
    jac = csr_matrix((values, p.indices, p.indptr), shape=(p.m, p.n))
    value, lower, upper = np.r_[c, x], np.r_[p.clower, p.lower], np.r_[p.cupper, p.upper]
    gradients = vstack([jac, eye(p.n, format="csr")], format="csr")
    primal = float(np.max(np.abs(value - np.clip(value, lower, upper))))

    def certificate(mu):
        lower_side = mu > 0
        bound = np.where(lower_side, lower, upper)
        finite = np.isfinite(bound)
        dual_sign = float(np.max(np.abs(mu[~finite]), initial=0))
        complementarity = float(np.max(np.abs(mu[finite] * (value[finite] - bound[finite])),
                                       initial=0))
        stationarity = float(np.max(np.abs(g - gradients.T @ mu)))
        return dict(stationarity=stationarity, complementarity=complementarity,
                    dual_sign=dual_sign, accepted=bool(primal <= PRIMAL_TOL
                        and stationarity <= STATIONARITY_TOL and complementarity <= 1e-6
                        and dual_sign <= 1e-8))

    if lam is not None and z is not None:
        cert = certificate(np.r_[lam, z])
        if cert["accepted"]:
            return dict(primal=primal, source="solver", **cert)
    if primal > 1e-4:
        return dict(primal=primal, source="solver", **certificate(np.r_[lam, z])) if (
            lam is not None and z is not None) else dict(primal=primal, accepted=False)
    active = (lower == upper) | (np.abs(value - lower) <= 1e-6 * (1 + np.abs(value))) | (
        np.abs(value - upper) <= 1e-6 * (1 + np.abs(value)))
    mu = np.zeros(len(value))
    if np.any(active):
        mu[active] = lsmr(gradients[active].T, g, atol=1e-12, btol=1e-12, maxiter=4*p.n)[0]
    return dict(primal=primal, source="least-squares", **certificate(mu))


def run(p, method, reuse, scaling, initial_scale, iterations, seconds):
    if method in {"dense", "partitioned"}:
        from sqp_craft import Options, SqpOptions, solve

        hessian = dict(hessian="partitioned", qp_reuse_workspace=reuse) if method == "partitioned" else {}
        options = SqpOptions(scaling=scaling, initial_hessian_scale=initial_scale,
                             max_iterations=iterations, max_callback_calls=12000,
                             time_limit_seconds=seconds, stationarity_tolerance=STATIONARITY_TOL,
                             **hessian)
        began = time.perf_counter()
        result = solve(p.spec(method == "partitioned"), solvers._SqpModel(p), p.x0, method="sqp",
                       options=Options(feasibility_tolerance=PRIMAL_TOL), method_options=options)
        m = result.multipliers
        record = dict(status=result.status, iterations=result.iterations, x=result.candidate,
                      message=result.message, seconds=time.perf_counter() - began,
                      functions=result.counts.functions, derivatives=result.counts.derivatives,
                      constraint_multipliers=m.constraints if m else None,
                      variable_multipliers=m.variables if m else None)
        for key in ("qp_solves", "qp_iterations", "qp_setups", "qp_updates", "qp_cold_retries",
                    "qp_setup_seconds", "qp_solve_seconds", "qp_conversion_seconds",
                    "bfgs_seconds", "callback_seconds", "qp_status"):
            record[key] = getattr(result.report, key)
        structure = getattr(result.report, "structure", None)
        if structure is not None:
            record["structure"] = {k: v for k, v in structure.items() if k != "elements"}
            for key in ("updates", "damped", "modified", "skipped", "resets"):
                record["structure"][key] = sum(e[key] for e in structure["elements"])
    else:
        solvers.MAX_ITERATIONS, solvers.TIME_LIMIT = iterations, seconds
        p.track_counts = method == "slsqp"
        record = solvers.run(method, p)
        p.track_counts = False
    if record.get("x") is not None:
        x = np.asarray(record["x"])
        record["check"] = check(p, x, record["constraint_multipliers"], record["variable_multipliers"])
        record["replay"] = p.replay(x)
        record["controls"] = p.unpack_controls(x).tolist()
        record["departure_excess"], record["arrival_mass"] = x[:3].tolist(), float(x[-1])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arcs", type=int, required=True)
    parser.add_argument("--floor", type=float, choices=(0.0, 0.02), default=0.02)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--seed", choices=("lambert", "offset"), default="lambert")
    parser.add_argument("--start-file", type=Path)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--scaling", choices=("none", "jacobian"), default="none",
                        help="none (default) keeps the model's nondimensional units; jacobian "
                        "is the adverse case that fails early on the large Lambert starts")
    parser.add_argument("--initial-scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--method", choices=("dense", "partitioned", "slsqp", "uno-filtersqp",
                                              "uno-ipopt", "derivatives"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = Transfer(args.arcs, args.floor, args.single, args.seed, args.start_file)
    if args.method == "derivatives":
        from sqp_craft import check_derivatives

        began = time.perf_counter()
        result = check_derivatives(p.spec(), p, p.x0, max_directions=p.n, steps=[1e-5, 1e-6],
                                   absolute_tolerance=2e-7, relative_tolerance=2e-5)
        record = dict(passed=result.passed, checked=result.checked, skipped=len(result.skipped),
                      issues=len(result.issues), seconds=time.perf_counter() - began)
    else:
        record = run(p, args.method, not args.no_reuse, args.scaling, args.initial_scale, args.iterations, args.seconds)
    record.update(arcs=p.arcs, segments=p.segments, n=p.n, m=p.m, nnz=p.nnz, floor=args.floor,
                  formulation="single" if args.single else "chain", solver=args.method,
                  seed=args.seed, reuse=args.method == "partitioned" and not args.no_reuse,
                  scaling=args.scaling, initial_scale=args.initial_scale, time_budget=args.seconds,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in
                      {"x", "constraint_multipliers", "variable_multipliers", "controls"}}))


if __name__ == "__main__":
    main()
