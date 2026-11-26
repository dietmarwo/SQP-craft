"""Run the sparse benchmark bridge and independently verify both solvers."""
from argparse import ArgumentParser
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tracemalloc

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, minimize

from fill import ORDERINGS, compare, eliminate, graph
from model import (
    ABS_TOL, OBJECTIVE_TOL, Branch, Fixture, Stage, aggregate_native_bounds, assemble,
    certificate, evaluate, forward_rollout, maximum, native_candidate, read_fixture,
    validate_support, write_fixture,
)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def invoke_upstream(executable, args):
    subprocess.run([str(executable), *map(str, args)], check=True, timeout=120)


def invoke_bridge(executable, input_path):
    with input_path.open("rb") as data:
        cp = subprocess.run([str(executable)], stdin=data, capture_output=True, timeout=120)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.decode("utf-8", errors="replace"))
    return json.loads(cp.stdout)


def toy(J, initial, constrained):
    weight = 1 / J
    N = 8
    A = np.array([[1.0, 0.1], [0.0, 0.98]])
    B = np.array([[0.005], [0.1]])
    H = weight * np.diag([3.0, 0.4, 0.2])
    HN = weight * np.diag([6.0, 0.8])
    g = -H @ [1.0, 0.0, 0.0]
    gN = -HN @ [1.0, 0.0]
    bound = 2.0 if constrained else np.inf
    branches = []
    for scale in ([0.9] if J == 1 else [0.9, 1.1]):
        dynamics = np.column_stack((A, scale * B))
        stages = [Stage(H, g, np.eye(3)[[2, 0]], np.full(2, -bound),
                        np.full(2, bound), dynamics, np.zeros(2))] * N
        stages.append(Stage(HN, gN, np.array([[1., 0.]]), np.array([-bound]), np.array([bound])))
        branches.append(Branch(np.asarray(initial), tuple(stages)))
    return Fixture(tuple(branches), 2, 1, 0.5 * (N * 3.0 + 6.0))


def assert_fixture_equal(first, second):
    require((first.J, first.N, first.nx, first.nu) == (second.J, second.N, second.nx, second.nu),
            "upstream round trip changed dimensions")
    for a, b in zip(first.branches, second.branches, strict=True):
        np.testing.assert_array_equal(a.initial, b.initial)
        for c, d in zip(a.stages, b.stages, strict=True):
            for name in ("hessian", "gradient", "selectors", "lower", "upper", "dynamics", "offset"):
                np.testing.assert_array_equal(getattr(c, name), getattr(d, name))


def relative_difference(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def mapping_checks(fixture, q):
    L = q.layout
    rng = np.random.default_rng(118)
    root_perturbation = np.zeros(L.n)
    root_perturbation[L.root] = np.linspace(0.03, 0.09, L.nu)
    for x in (np.zeros(L.n), rng.normal(size=L.n), root_perturbation):
        obj, residual, gradient, _ = evaluate(fixture, L, x)
        require(relative_difference(obj, q.cost(x)) < 1e-13, "stage/CSR objective mismatch")
        np.testing.assert_allclose(residual, q.jacobian @ x - q.rhs, atol=1e-12, rtol=0)
        np.testing.assert_allclose(gradient, q.product(x) + q.gradient, atol=1e-12, rtol=0)
    if L.J < 2:
        return
    x = root_perturbation.copy()
    x[L.x(0, 1)] = 0.125
    _, changed, _, _ = evaluate(fixture, L, x)
    _, original, _, _ = evaluate(fixture, L, root_perturbation)
    require(np.any(changed[:L.N * L.nx] != original[:L.N * L.nx]), "ineffective branch test")
    np.testing.assert_array_equal(changed[L.N * L.nx:], original[L.N * L.nx:])
    permuted = replace(fixture, branches=fixture.branches[::-1])
    reordered = assemble(permuted)
    reversed_x = np.r_[x[:L.root.start].reshape(L.J, L.branch_width)[::-1].ravel(), x[L.root]]
    np.testing.assert_allclose(reordered.cost(reversed_x), q.cost(x), rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(
        (reordered.jacobian @ reversed_x - reordered.rhs).reshape(L.J, L.N, L.nx),
        changed.reshape(L.J, L.N, L.nx)[::-1], rtol=0, atol=1e-12,
    )
    corrupted = q.jacobian.copy()
    corrupted.indices[0] = L.x(1, 1).start
    try:
        validate_support(replace(q, jacobian=corrupted))
    except ValueError:
        return
    raise ValueError("off-branch Jacobian entry passed eligibility")


def compare_with_upstream(fixture, q, upstream, generic):
    point, lam, y, root, initial = native_candidate(fixture, q, upstream)
    direct = certificate(fixture, q, point, lam, native_duals=y)
    z = aggregate_native_bounds(q, y)
    reduced = certificate(fixture, q, point, lam, box_dual=z)
    reference = certificate(fixture, q, generic["x"], generic["lambda"], box_dual=generic["z"])
    objective_difference = relative_difference(direct["objective"], reference["objective"])
    L = q.layout
    generic_x = np.asarray(generic["x"])
    labels = L.labels()
    is_state = np.array([label[2] == "state" for label in labels])
    state_names = [label for label in labels if label[2] == "state"]
    input_names = [label for label in labels if label[2] != "state"]
    delta = point - generic_x
    rollout_delta = forward_rollout(fixture, L, point) - forward_rollout(fixture, L, generic_x)
    result = {
        "status": upstream["status"], "original": direct, "after_root_bound_aggregation": reduced,
        "shared_input_copies": root, "fixed_initial_state_copies": initial,
        "relative_objective_difference": objective_difference,
        "maximum_state_difference": maximum(delta[is_state], state_names),
        "maximum_control_difference": maximum(delta[~is_state], input_names),
        "maximum_rollout_difference": maximum(rollout_delta[is_state], state_names),
        "passed": bool(upstream["status"] == "Converged" and reference["passed"]
                       and direct["passed"] and reduced["passed"]
                       and root["value"] <= ABS_TOL and initial["value"] <= ABS_TOL
                       and objective_difference <= OBJECTIVE_TOL),
    }
    return result


def negative_solution_checks(fixture, q, upstream, generic):
    j = len(upstream["branches"]) - 1
    data = {"schema": upstream["schema"], "shared_input": upstream["shared_input"],
            "branches": [dict(br, trajectory=br["trajectory"].copy()) for br in upstream["branches"]]}
    data["branches"][j]["trajectory"][fixture.nx] += 1e-3
    require(native_candidate(fixture, q, data)[3]["value"] > ABS_TOL,
            "shared input discrepancy went undetected")
    wrong = np.asarray(generic["z"]).copy()
    i = int(np.argmax(abs(wrong)))
    require(abs(wrong[i]) > 1e-5, "published fixture has no significant box multiplier")
    wrong[i] *= -1
    require(not certificate(fixture, q, generic["x"], generic["lambda"], box_dual=wrong)["passed"],
            "checker accepted a flipped active box multiplier")
    wrong_dynamics = np.asarray(generic["lambda"]).copy()
    wrong_dynamics[int(np.argmax(abs(wrong_dynamics)))] *= -1
    require(not certificate(fixture, q, generic["x"], wrong_dynamics, box_dual=generic["z"])["passed"],
            "checker accepted a flipped dynamics multiplier")
    return {"root_copy_rejected": True, "bound_sign_rejected": True, "dynamics_sign_rejected": True}


def check_fill_on_toy():
    q = assemble(toy(2, [0.4, -0.1], True))
    pattern = graph(q)
    records = compare(q)
    for name in ORDERINGS:
        record = records[name]
        dense = np.eye(len(pattern), dtype=bool)
        for i, row in enumerate(pattern):
            dense[i, list(row)] = True
        dense = dense[np.ix_(record["permutation"], record["permutation"])]
        count = len(pattern)
        for i in range(len(pattern)):
            neighbors = np.flatnonzero(dense[i, i + 1:]) + i + 1
            count += len(neighbors)
            dense[np.ix_(neighbors, neighbors)] = True
        require(count == record["factor_lower_nnz_including_diagonal"], "symbolic count mismatch")
    star = [{1, 2, 3}, {0}, {0}, {0}]
    require(eliminate(star, [1, 2, 3, 0])["factor_lower_nnz_including_diagonal"] == 7,
            "star elimination mismatch")
    require(eliminate(star, [0, 1, 2, 3])["factor_lower_nnz_including_diagonal"] == 10,
            "star fill mismatch")


def run_toys(args):
    results = []
    starts = ([0.4, -0.1], [0.2, -0.15], [-0.3, 0.2])
    for J in (1, 2):
        for s, start in enumerate(starts):
            for bounded in (False, True):
                case = f"toy-J{J}-start{s}-{'bounded' if bounded else 'free'}"
                fixture = toy(J, start, bounded)
                q = assemble(fixture)
                mapping_checks(fixture, q)
                path = args.output / f"{case}.qp"
                q.write(path)
                raw = invoke_bridge(args.binary, path)
                check = certificate(fixture, q, raw["x"], raw["lambda"], box_dual=raw["z"])
                # Dense matrices are limited to the existing 24/47-variable reference cases.
                E = q.jacobian.toarray()
                if bounded:
                    oracle = minimize(q.cost, np.zeros(q.layout.n),
                                      jac=lambda x: q.product(x) + q.gradient,
                                      bounds=Bounds(q.lower, q.upper),
                                      constraints=LinearConstraint(E, q.rhs, q.rhs),
                                      method="SLSQP", options={"ftol": 1e-12, "maxiter": 400})
                    success, reference = oracle.success, oracle.x
                else:
                    matrix = np.block([[np.diag(q.lower_hessian.diagonal()), E.T],
                                       [E, np.zeros((q.layout.m, q.layout.m))]])
                    reference = np.linalg.solve(matrix, np.r_[-q.gradient, q.rhs])[:q.layout.n]
                    success = True
                discrepancy = relative_difference(check["objective"], q.cost(reference))
                passed = bool(success and raw["status"] == "optimal" and check["passed"]
                              and discrepancy <= OBJECTIVE_TOL and check["forward_rollout"]["value"] <= ABS_TOL)
                record = {"name": case, "passed": passed, "oracle": "SLSQP" if bounded else "direct KKT",
                          "certificate": check, "relative_objective_difference": discrepancy}
                if args.upstream and J == 2 and s == 0:
                    incoming = args.output / f"{case}.fixture"
                    round_trip = args.output / f"{case}-upstream.fixture"
                    native_solution = args.output / f"{case}-upstream.json"
                    write_fixture(incoming, fixture)
                    invoke_upstream(args.upstream, ["import", incoming, round_trip, native_solution])
                    assert_fixture_equal(fixture, read_fixture(round_trip))
                    paired = compare_with_upstream(fixture, q, json_file(native_solution), raw)
                    record["upstream_round_trip"] = paired
                    record["passed"] = passed and paired["passed"]
                results.append(record)
    check_fill_on_toy()
    save(args.output / "results.json", results)
    return all(row["passed"] for row in results)


def sizes(q):
    L = q.layout
    record = {}
    for name, matrix in (("hessian_lower", q.lower_hessian), ("jacobian", q.jacobian)):
        record[name] = {
            "shape": list(matrix.shape), "structural_nnz": matrix.nnz,
            "numeric_nnz": int(np.count_nonzero(matrix.data)),
            "csr_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
            "largest_row": int(np.max(np.diff(matrix.indptr))),
        }
    record["n"] = L.n
    record["m"] = L.m
    record["root_width"] = L.nu
    record["two_sided_boxes"] = int(np.count_nonzero(np.isfinite(q.lower) & np.isfinite(q.upper)))
    record["dense_jacobian_bytes_avoided"] = L.n * L.m * np.dtype("f8").itemsize
    return record


def audit_size_scaling(fixture, q):
    more_branches = assemble(replace(fixture, branches=fixture.branches * 2))
    longer = assemble(replace(fixture, branches=tuple(
        replace(b, stages=b.stages[:-1] * 2 + b.stages[-1:]) for b in fixture.branches
    )))
    L = q.layout
    require(more_branches.layout.n == 2 * L.n - L.nu and more_branches.layout.m == 2 * L.m
            and more_branches.jacobian.nnz == 2 * q.jacobian.nnz, "branch size scaling")
    extra = 0
    for branch in fixture.branches:
        extra += sum(np.count_nonzero(s.dynamics) + L.nx for s in branch.stages[:-1])
    require(longer.layout.n == L.n + L.J * L.N * (L.nx + L.nu)
            and longer.layout.m == 2 * L.m and longer.jacobian.nnz == q.jacobian.nnz + extra,
            "stage size scaling")
    return {"published": sizes(q), "twice_branches": sizes(more_branches), "twice_horizon": sizes(longer)}


def exact_step(fixture, q, verdict):
    from sqp_craft import Options, ProblemSpec, SqpOptions, solve

    E, H = q.jacobian, q.lower_hessian
    L = q.layout
    base = forward_rollout(fixture, L, np.zeros(L.n))
    require(np.any(base != 0) and np.all((q.lower <= base) & (base <= q.upper)),
            "exact-step start is not a nonzero, box-feasible rollout")

    class Model:
        def __init__(self):
            self.hessian_points = []

        def functions(self, x):
            point = np.asarray(x)
            return q.cost(point), (E @ point).tolist()

        def derivatives(self, x):
            return (q.product(np.asarray(x)) + q.gradient).tolist(), E.data.tolist()

        def lagrangian_hessian(self, x, _multipliers):
            self.hessian_points.append(list(x))
            return H.data.tolist()

    spec = ProblemSpec(
        lower=q.lower, upper=q.upper, constraint_lower=q.rhs, constraint_upper=q.rhs,
        jacobian_indptr=E.indptr, jacobian_indices=E.indices, linear_rows=range(L.m),
        hessian_indptr=H.indptr, hessian_indices=H.indices,
    )
    model = Model()
    result = solve(
        spec, model, base, method="sqp", options=Options(feasibility_tolerance=ABS_TOL),
        method_options=SqpOptions(hessian="exact", qp_solver="clarabel", scaling="none",
                                  max_iterations=1, record_qp_limit=4, qp_tolerance=1e-11,
                                  stationarity_tolerance=ABS_TOL),
    )
    report = result.report
    require(report.qp_trace and model.hessian_points, f"exact SQP emitted no step: {result.message}")
    require(result.candidate is not None and result.multipliers is not None, "no exact SQP candidate")
    trace = report.qp_trace[0]
    start = np.asarray(model.hessian_points[0])
    np.testing.assert_array_equal(start, base)
    for matrix, name in ((E, "matrix"), (H, "hessian_lower")):
        tr = trace[name]
        require((tr["rows"], tr["cols"]) == matrix.shape, f"{name} trace shape")
        for key in ("indices", "indptr"):
            np.testing.assert_array_equal(tr[key], getattr(matrix, key))
        np.testing.assert_array_equal(tr["values"], matrix.data)
    expected = {"gradient": q.gradient + q.product(start), "lower": q.lower - start,
                "upper": q.upper - start, "row_lower": q.rhs - E @ start, "row_upper": q.rhs - E @ start}
    differences = {}
    for key, check in expected.items():
        values = np.asarray(trace[key])
        np.testing.assert_allclose(values, check, atol=1e-12, rtol=0)
        finite = np.isfinite(check)
        differences[key] = float(max(abs(values[finite] - check[finite]), default=0))
    actual = certificate(fixture, q, result.candidate, result.multipliers.constraints,
                         box_dual=result.multipliers.variables)
    structure = report.structure
    require(structure is not None and structure["exact_hessian_calls"] >= 1, "no exact callback recorded")
    difference = relative_difference(verdict["objective"], actual["objective"])
    details = {
        "status": result.status, "message": result.message, "backend": report.qp_backend,
        "accepted_steps": result.iterations, "qp_solves": report.qp_solves,
        "maximum_start_coordinate": float(max(abs(base))), "start_equality": float(max(abs(E @ base - q.rhs))),
        "linear_projection": report.linear_projection,
        "scale_deviations": max(abs(np.r_[report.variable_scales, report.row_scales] - 1), default=0),
        "max_recorded_qp_coefficient_differences": differences,
        "trace_count": len(report.qp_trace), "structure": structure,
        "restore_attempts": report.restoration_attempts, "qp_cold_retries": report.qp_cold_retries,
        "certificate": actual, "relative_objective_difference": difference,
    }
    details["passed"] = bool(actual["passed"] and difference <= OBJECTIVE_TOL
                             and result.iterations == 1 and structure["diagonal_shifts"] == 0)
    return details


def run_published(args):
    out = args.output
    fixture_path, reference_path = out / "spring-mass.fixture", out / "fancyqpalm.json"
    if args.published:
        invoke_upstream(args.upstream, ["published", fixture_path, reference_path])
    else:
        shutil.copyfile(args.fixture, fixture_path)
        if args.reference_solution:
            shutil.copyfile(args.reference_solution, reference_path)
    fixture = read_fixture(fixture_path)
    tracemalloc.start()
    try:
        q = assemble(fixture)
        _, assembly_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    L = q.layout
    require((L.J, L.N, L.nx, L.nu, L.n, L.m) == (4, 32, 24, 6, 3822, 3072),
            "unexpected Figure 3 dimensions")
    require(np.count_nonzero(np.isfinite(q.lower) & np.isfinite(q.upper)) == 2286,
            "unexpected position/control bound count")
    np.testing.assert_array_equal(q.lower_hessian.diagonal()[L.root], np.full(6, 4.0))
    require(q.jacobian.nnz <= 92928 and q.lower_hessian.nnz == L.n,
            "unexpected Hessian or Jacobian structure")
    mapping_checks(fixture, q)
    q.write(out / "spring-mass.qp")
    raw = invoke_bridge(args.binary, out / "spring-mass.qp")
    save(out / "clarabel.json", raw)
    result = {"status": raw["status"], "accepted_by_qp_model": raw.get("accepted_by_qp_model", False)}
    verdict = None
    if "x" in raw:
        verdict = certificate(fixture, q, raw["x"], raw["lambda"], box_dual=raw["z"])
        result["clarabel"] = verdict
    if reference_path.exists() and verdict:
        upstream = json_file(reference_path)
        result["fancyqpalm"] = compare_with_upstream(fixture, q, upstream, raw)
        result["negative_checks"] = negative_solution_checks(fixture, q, upstream, raw)
    else:
        result["fancyqpalm"] = {"status": "missing comparator or generic candidate", "passed": False}
    if args.exact_step and verdict:
        result["exact_sqp"] = exact_step(fixture, q, verdict)
    save(out / "residuals.json", result)

    check_fill_on_toy()
    fill = compare(q)
    sides = int(np.count_nonzero(np.isfinite(q.lower)) + np.count_nonzero(np.isfinite(q.upper)))
    fill["clarabel_native"] = {
        "order": "QDLDL AMD (Clarabel 0.11.1 default build)", "presolve": "enabled",
        "box_sides": sides, "zero_cone_dynamics_rows": L.m,
        "kkt_dimension_from_finite_cone_rows": L.n + L.m + sides,
        "off_diagonal_factor_nnz": raw["qp_factor_nnz"],
        "budget_estimate_factor_nnz": raw["kkt_factor_nnz"],
        "hessian_factor_nnz": raw["hessian_factor_nnz"],
        "cold_retries": raw["cold_retries"],
    }
    save(out / "fill.json", fill)
    audit = audit_size_scaling(fixture, q)
    audit["python_traced_assembly_peak_bytes"] = assembly_peak
    audit["wire_bytes"] = (out / "spring-mass.qp").stat().st_size
    audit["bridge_csr_payload_bytes"] = raw["csr_payload_bytes"]
    audit["stage_input_bytes"] = fixture_path.stat().st_size
    audit["largest_input_stage_shapes"] = {
        "dynamics": [L.nx, L.nx + L.nu],
        "selectors": list(fixture.branches[0].stages[0].selectors.shape),
        "cost": [L.nx + L.nu, L.nx + L.nu],
    }
    audit["inverse_variable_layout"] = L.labels()
    audit["bound_origin"] = q.bound_origin
    save(out / "allocations.json", audit)
    save(out / "manifest.json", {
        "fixture_sha256": digest(fixture_path), "upstream_solution_sha256":
            digest(reference_path) if reference_path.exists() else None,
        "sparse_qp_sha256": digest(out / "spring-mass.qp"), "bridge_sha256": digest(args.binary),
        "upstream_binary_sha256": digest(args.upstream) if args.upstream else None,
        "fixture_description": "FanCyqlone Fig. 3, wang-boyd-2008, M=12, N=32, J=4, seed=0",
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    })
    return bool(verdict and verdict["passed"] and result["accepted_by_qp_model"]
                and raw["status"] == "optimal" and result["fancyqpalm"]["passed"]
                and (not args.exact_step or result.get("exact_sqp", {}).get("passed", False)))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=lambda p: Path(p).resolve())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=lambda p: Path(p).resolve(),
                        help="instrumented spring-mass-fixture executable")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--published", action="store_true", help="generate the Figure 3 fixture and solve it")
    source.add_argument("--fixture", type=Path, help="replay the exported Figure 3 fixture")
    parser.add_argument("--reference-solution", type=Path, help="previous FanCyQPALM solution for --fixture")
    parser.add_argument("--exact-step", action="store_true", help="exercise the installed SQP-craft Python API")
    args = parser.parse_args()
    if args.published and not args.upstream:
        parser.error("--published requires --upstream")
    if args.reference_solution and not args.fixture:
        parser.error("--reference-solution requires --fixture")
    if args.exact_step and not (args.fixture or args.published):
        parser.error("--exact-step requires the published fixture")
    args.output.mkdir(parents=True, exist_ok=False)
    passed = run_published(args) if args.published or args.fixture else run_toys(args)
    print(f"{'Passed' if passed else 'Incomplete or failed'} — {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
