"""Recheck paired independent validation without reinterpreting failures as speedups."""

import argparse, json, hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("docs/performance-30/data"))
    a = p.parse_args()
    expected = {
        f"baseline-{backend}-seed{seed}.json"
        for backend in ["active-set", "clarabel"]
        for seed in [1, 2, 3, 4, 5, 2026]
    }
    expected |= {
        "baseline-elastic-seed23.json",
        "baseline-independent-development.json",
        "baseline-independent-heldout.json",
        "baseline-variants.json",
    }
    actual = {p.name for p in (a.data / "stress").glob("baseline-*.json")}
    assert actual == expected, dict(
        missing=sorted(expected - actual), unexpected=sorted(actual - expected)
    )
    environment = json.loads((a.data / "environment.json").read_text())
    root = Path(__file__).resolve().parents[1]
    for name, sha in environment["callbacks_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    losses = []
    changes = []
    pairs = 0
    objective = 0.0
    iterations = 0
    for path in sorted((a.data / "stress").glob("baseline-*.json")):
        other = path.with_name(path.name.replace("baseline-", "final-", 1))
        if not other.exists():
            raise RuntimeError(f"missing pair: {other}")
        b = json.loads(path.read_text())
        f = json.loads(other.read_text())
        if path.name == "baseline-variants.json":
            groups = [
                (family + "/" + variant, rows, f[family][variant])
                for family, vs in b.items()
                for variant, rows in vs.items()
            ]
        else:
            assert b["version"] == "0.14.0" and f["version"] == "0.15.0"
            if "seed" in b:
                assert b["cases"] == f["cases"] == 300
                keys = {
                    (case, mode)
                    for case in range(300)
                    for mode in ["cold", "reuse", "restoration_off"]
                }
                assert {(r["case"], r["mode"]) for r in b["records"]} == keys
                assert {(r["case"], r["mode"]) for r in f["records"]} == keys
            assert (
                not b.get("errors")
                and not f.get("errors")
                and not b.get("false_claims")
                and not f.get("false_claims")
            )
            groups = [("", b["records"], f["records"])]
        for group, br, fr in groups:
            assert len(br) == len(fr)
            for i, (base, new) in enumerate(zip(br, fr)):
                pairs += 1
                keys = ["case", "mode", "key", "family", "size", "backend"]
                assert tuple(base.get(k) for k in keys) == tuple(
                    new.get(k) for k in keys
                )
                if (
                    (base.get("accepted") and not new.get("accepted"))
                    or (base["status"] == "converged" and new["status"] != "converged")
                    or (base.get("reference") is True and new.get("reference") is False)
                ):
                    losses.append(
                        dict(
                            file=path.name,
                            group=group,
                            index=i,
                            baseline=base,
                            final=new,
                        )
                    )
                if base["status"] != new["status"] or base.get("accepted") != new.get(
                    "accepted"
                ):
                    changes.append(
                        dict(
                            file=path.name,
                            group=group,
                            index=i,
                            baseline_status=base["status"],
                            final_status=new["status"],
                            baseline_accepted=base.get("accepted"),
                            final_accepted=new.get("accepted"),
                        )
                    )
                iterations += base.get("iterations") != new.get("iterations")
                if (
                    base.get("check")
                    and new.get("check")
                    and base.get("accepted")
                    and new.get("accepted")
                ):
                    objective = max(
                        objective,
                        abs(base["check"]["objective"] - new["check"]["objective"]),
                    )
    assert pairs == 17372, pairs
    result = dict(
        paired_solves=pairs,
        total_solver_calls=2 * pairs,
        losses=losses,
        status_or_acceptance_changes=changes,
        outer_iteration_changes=iterations,
        max_accepted_objective_difference=objective,
    )
    (a.data / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if losses:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
