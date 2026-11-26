"""The fan-QP study on the committed inputs of the published spring-mass case.

Without the Rust bridge the tests check the stage model, the reference
solution and the symbolic fill. With `FAN_BRIDGE` set to the built
`fan_bridge` example they also run the toy cases and the offline replay.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "bench/fan-qp"
RESULTS = STUDY / "results/2026-11-26"
sys.path.insert(0, str(STUDY))

import fill  # noqa: E402
import model  # noqa: E402

LIMITS = dict(equality=1e-8, inequality=1e-8, stationarity=1e-8, dual_sign=1e-8, complementarity=1e-6)


def load(name):
    return json.loads(gzip.decompress((RESULTS / name).read_bytes()))


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    path = tmp_path_factory.mktemp("fan") / "spring-mass.fixture"
    path.write_bytes(gzip.decompress((RESULTS / "spring-mass.fixture.gz").read_bytes()))
    fixture = model.read_fixture(path)
    return fixture, model.assemble(fixture)


def same(value, recorded):
    """Residuals near rounding level differ between BLAS kernels; magnitudes do not."""
    return value == pytest.approx(recorded, rel=1e-6, abs=1e-11)


def test_committed_inputs_are_the_recorded_ones(case):
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for name, key in (("spring-mass.fixture.gz", "fixture_sha256"),
                      ("fancyqpalm.json.gz", "upstream_solution_sha256")):
        data = gzip.decompress((RESULTS / name).read_bytes())
        assert hashlib.sha256(data).hexdigest() == manifest[key]
    _, q = case
    L = q.layout
    assert (L.J, L.N, L.nx, L.nu, L.n, L.m) == (4, 32, 24, 6, 3822, 3072)
    assert (q.jacobian.nnz, q.lower_hessian.nnz) == (92928, 3822)
    assert np.count_nonzero(np.isfinite(q.lower) & np.isfinite(q.upper)) == 2286
    assert q.offset == pytest.approx(90.45122306451248, rel=1e-14)


def test_reference_solution_passes_the_original_stage_checks(case):
    fixture, q = case
    recorded = json.loads((RESULTS / "residuals.json").read_text())["fancyqpalm"]
    point, lam, native, root, initial = model.native_candidate(fixture, q, load("fancyqpalm.json.gz"))
    assert root["value"] == initial["value"] == 0.0
    direct = model.certificate(fixture, q, point, lam, native_duals=native)
    merged = model.certificate(fixture, q, point, lam,
                               box_dual=model.aggregate_native_bounds(q, native))
    for check, record in ((direct, recorded["original"]),
                          (merged, recorded["after_root_bound_aggregation"])):
        assert check["passed"] and record["passed"]
        assert check["objective"] == pytest.approx(record["objective"], rel=1e-13)
        for name, limit in LIMITS.items():
            assert check[name]["value"] <= limit
            assert same(check[name]["value"], record[name]["value"])
    assert direct["objective"] == pytest.approx(1778.031254158380, rel=1e-13)
    # The checker must not accept the same point with a multiplier of the wrong sign.
    wrong = lam.copy()
    wrong[int(np.argmax(abs(wrong)))] *= -1
    assert not model.certificate(fixture, q, point, wrong, native_duals=native)["passed"]
    shifted = point.copy()
    shifted[q.layout.root.start] += 1e-3
    assert not model.certificate(fixture, q, shifted, lam, native_duals=native)["passed"]


def test_symbolic_fill_matches_the_record(case):
    _, q = case
    recorded = load("fill.json.gz")
    fresh = fill.compare(q)
    assert fresh["graph_sha256"] == recorded["graph_sha256"]
    assert fresh["kkt_lower_nnz_including_diagonal"] == 99822
    for name in fill.ORDERINGS:
        assert fresh[name] == recorded[name]
    counts = {name: fresh[name]["factor_lower_nnz_including_diagonal"] for name in fill.ORDERINGS}
    assert counts == dict(generic_minimum_degree=204333, root_last_minimum_degree=204333,
                          root_last=240081, root_first=243522)
    # Hand counts: a star eliminated leaves first has no fill, centre first fills in.
    star = [{1, 2, 3}, {0}, {0}, {0}]
    assert fill.eliminate(star, [1, 2, 3, 0])["fill_edges"] == 0
    assert fill.eliminate(star, [0, 1, 2, 3])["fill_edges"] == 3
    with pytest.raises(ValueError, match="permutation"):
        fill.eliminate(star, [0, 1, 2, 2])


def bridge():
    path = os.environ.get("FAN_BRIDGE")
    if not path:
        pytest.skip("set FAN_BRIDGE to the built fan_bridge example")
    return str(Path(path).resolve())


def run_study(output, *arguments):
    return subprocess.run([sys.executable, str(STUDY / "run.py"), "--binary", bridge(),
                           "--output", str(output), *arguments],
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})


def test_toy_cases_pass_through_the_bridge(tmp_path):
    done = run_study(tmp_path / "toys")
    assert done.returncode == 0, done.stdout + done.stderr
    rows = json.loads((tmp_path / "toys/results.json").read_text())
    assert len(rows) == 12 and all(row["passed"] for row in rows)


def test_offline_replay_reproduces_the_record(tmp_path):
    exact = []
    try:
        import sqp_craft  # noqa: F401
        exact = ["--exact-step"]
    except ImportError:
        pass
    out = tmp_path / "replay"
    done = run_study(out, "--fixture", str(RESULTS / "spring-mass.fixture.gz"),
                     "--reference-solution", str(RESULTS / "fancyqpalm.json.gz"), *exact)
    assert done.returncode == 0, done.stdout + done.stderr
    fresh = json.loads((out / "residuals.json").read_text())
    recorded = json.loads((RESULTS / "residuals.json").read_text())
    assert fresh["status"] == recorded["status"] == "optimal" and fresh["accepted_by_qp_model"]
    assert fresh["negative_checks"] == recorded["negative_checks"]
    pairs = [(fresh["clarabel"], recorded["clarabel"]),
             (fresh["fancyqpalm"]["original"], recorded["fancyqpalm"]["original"])]
    if exact:
        assert fresh["exact_sqp"]["passed"] and fresh["exact_sqp"]["structure"]["diagonal_shifts"] == 0
        pairs.append((fresh["exact_sqp"]["certificate"], recorded["exact_sqp"]["certificate"]))
    for check, record in pairs:
        assert check["passed"]
        assert check["objective"] == pytest.approx(record["objective"], rel=1e-11)
        for name, limit in LIMITS.items():
            assert check[name]["value"] <= limit
    assert fresh["fancyqpalm"]["relative_objective_difference"] <= 1e-7
    assert json.loads((out / "fill.json").read_text()) == load("fill.json.gz")
    manifest = json.loads((out / "manifest.json").read_text())
    stored = json.loads((RESULTS / "manifest.json").read_text())
    for key in ("fixture_sha256", "upstream_solution_sha256", "sparse_qp_sha256"):
        assert manifest[key] == stored[key]
