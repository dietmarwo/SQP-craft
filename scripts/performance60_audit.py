"""Fail-closed completeness and numerical comparison of three-way solver calls."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def flattened(data, kind):
    payload = data["payload"]
    if kind == "variants":
        assert set(payload) == {"baseline", "maratos", "linear", "mpcc"}
        out = []
        for family, count in [
            ("baseline", 13),
            ("maratos", 200),
            ("linear", 600),
            ("mpcc", 600),
        ]:
            assert set(payload[family]) == {
                "default",
                "soc",
                "augmented",
                "scale-jacobian",
            }
            for variant in ["default", "soc", "augmented", "scale-jacobian"]:
                rows = payload[family][variant]
                assert len(rows) == count
                out += [(family + "/" + variant, row) for row in rows]
        return out
    assert not payload.get("errors") and not payload.get("false_claims")
    if "cases" in payload:
        count = payload["cases"]
        assert {(r["case"], r["mode"]) for r in payload["records"]} == {
            (i, m) for i in range(count) for m in ["cold", "reuse", "restoration_off"]
        }
    return [("", row) for row in payload["records"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    m = json.loads((a.data / "manifest.json").read_text())
    root = Path(__file__).resolve().parents[1]
    expected_calls = 920 if m["quick"] else 17372
    assert m["expected_calls_per_variant"] == expected_calls
    expected = {
        f"{label}-{group}.json"
        for label in ["baseline", "serial", "parallel"]
        for group in m["groups"]
    }
    actual = {
        p.name
        for p in a.data.glob("*.json")
        if p.name != "manifest.json" and not p.name.endswith(".payload.json")
    }
    assert actual == expected, dict(
        missing=sorted(expected - actual), unexpected=sorted(actual - expected)
    )
    for name, sha in m["callback_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    changes = []
    losses = []
    false = []
    iterations = 0
    objective = 0.0
    point = 0.0
    comparisons = 0
    calls = 0
    converged = collections.Counter()
    extensions = {k: set() for k in m["labels"]}
    for group, definition in m["groups"].items():
        by_label = {}
        flat = {}
        for label, width in m["labels"].items():
            data = json.loads((a.data / f"{label}-{group}.json").read_text())
            by_label[label] = data
            assert data["threads"] == width and data["version"] == (
                m["baseline_version"] if label == "baseline" else m["current_version"]
            )
            assert (
                data["observed_solver_calls"]
                == len(data["journal"])
                == definition["expected_calls"]
            )
            flat[label] = flattened(data, definition["kind"])
            assert len(flat[label]) == definition["expected_calls"]
            extensions[label].add(data["extension_sha256"])
            calls += len(data["journal"])
            for i, (_, row) in enumerate(flat[label]):
                j = data["journal"][i]
                assert (
                    j["index"] == i
                    and j["status"] == row["status"]
                    and j["iterations"] == row["iterations"]
                )
                if j["status"] == "converged":
                    converged[label] += 1
                    if not j["solver_certificate_accepted"]:
                        false.append([label, group, i])
                if j["backend"].startswith("goldfarb-idnani "):
                    assert j["backend"].endswith(data["version"])
        for left, right in [("baseline", "serial"), ("serial", "parallel")]:
            for i, ((bg, b), (fg, f)) in enumerate(zip(flat[left], flat[right])):
                comparisons += 1
                assert bg == fg and tuple(
                    b.get(k)
                    for k in ["case", "mode", "key", "family", "size", "backend"]
                ) == tuple(
                    f.get(k)
                    for k in ["case", "mode", "key", "family", "size", "backend"]
                )
                if b["status"] != f["status"] or b.get("accepted") != f.get("accepted"):
                    changes.append([left, right, group, i, b["status"], f["status"]])
                if (
                    (b["status"] == "converged" and f["status"] != "converged")
                    or (b.get("accepted") and not f.get("accepted"))
                    or (b.get("reference") is True and f.get("reference") is False)
                ):
                    losses.append([left, right, group, i])
                bj = by_label[left]["journal"][i]
                fj = by_label[right]["journal"][i]
                iterations += bj["iterations"] != fj["iterations"]
                if (
                    bj["solver_certificate_accepted"]
                    and fj["solver_certificate_accepted"]
                ):
                    objective = max(objective, abs(bj["objective"] - fj["objective"]))
                    assert len(bj["candidate"]) == len(fj["candidate"])
                    point = max(
                        point,
                        max(
                            (
                                abs(x - y)
                                for x, y in zip(bj["candidate"], fj["candidate"])
                            ),
                            default=0.0,
                        ),
                    )
    assert calls == 3 * expected_calls and comparisons == 2 * expected_calls
    assert (
        all(len(values) == 1 for values in extensions.values())
        and extensions["serial"] == extensions["parallel"]
    )
    result = dict(
        total_solver_calls=calls,
        paired_comparisons=comparisons,
        calls_per_variant=expected_calls,
        lost_acceptance_or_convergence=losses,
        status_or_acceptance_changes=changes,
        false_convergence_claims=false,
        outer_iteration_changes=iterations,
        max_accepted_objective_difference=objective,
        max_accepted_point_difference=point,
        converged=dict(converged),
        extension_sha256={k: next(iter(v)) for k, v in extensions.items()},
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert not losses and not false and not changes, result


if __name__ == "__main__":
    main()
