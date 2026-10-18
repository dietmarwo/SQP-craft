"""Regenerate exact tables and optionally the checkpoint figure from raw JSON."""

import argparse, csv, json, math, statistics
from pathlib import Path


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("docs/performance-30"))
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    D = a.root / "data"
    sections = []
    primary = json.loads((D / "final-common.json").read_text())
    rows = []
    for r in primary["summary"]:
        rows.append(
            [
                r["kind"],
                r["family"],
                r["size"],
                f"{r['baseline']*1000:.4f}",
                f"{r['final']*1000:.4f}",
                f"{r['speedup']:.3f}",
            ]
        )
    sections += [
        "## Common pinned baseline/final comparison",
        "",
        table(["Kind", "Fixture", "Size", "Baseline ms", "Final ms", "Speedup"], rows),
        "",
    ]
    hold = json.loads((D / "holdout-native.json").read_text())
    rows = []
    for family, size in hold["cases"]:
        for backend in ["active-set", "clarabel"]:
            b = next(
                r
                for r in hold["summary"]
                if (r["family"], r["size"], r["build"], r["backend"])
                == (family, size, "baseline", backend)
            )
            f = next(
                r
                for r in hold["summary"]
                if (r["family"], r["size"], r["build"], r["backend"])
                == (family, size, "final", backend)
            )
            assert b["passed"] == f["passed"] == hold["repeats"]
            rows.append(
                [
                    family,
                    size,
                    backend,
                    f"{b['seconds']*1000:.3f}",
                    f"{f['seconds']*1000:.3f}",
                    f"{b['seconds']/f['seconds']:.3f}",
                ]
            )
    sections += [
        "## Frozen held-out sizes",
        "",
        table(
            ["Fixture", "Size", "Backend", "Baseline ms", "Final ms", "Speedup"], rows
        ),
        "",
    ]
    both = json.loads((D / "final-both-backends.json").read_text())
    rows = []
    for family, size in [
        ("orbit", 20),
        ("orbit", 40),
        ("separable", 128),
        ("restoration", 64),
    ]:
        for backend in ["active", "clarabel"]:
            for mode in ["reuse", "cold"]:
                b = next(
                    r
                    for r in both["summary"]
                    if (r["family"], r["size"], r["build"], r["mode"])
                    == (family, size, "baseline-" + backend, mode)
                )
                f = next(
                    r
                    for r in both["summary"]
                    if (r["family"], r["size"], r["build"], r["mode"])
                    == (family, size, "final-" + backend, mode)
                )
                assert b["iterations"] == f["iterations"]
                rows.append(
                    [
                        family,
                        size,
                        backend,
                        mode,
                        f"{b['seconds']*1000:.3f}",
                        f"{f['seconds']*1000:.3f}",
                        f"{b['seconds']/f['seconds']:.3f}",
                        f"{b['bfgs_seconds']*1000:.3f}",
                        f"{f['bfgs_seconds']*1000:.3f}",
                    ]
                )
    sections += [
        "## Cold/reuse and both backends",
        "",
        table(
            [
                "Fixture",
                "Size",
                "Backend",
                "QP mode",
                "Baseline ms",
                "Final ms",
                "Speedup",
                "Base BFGS ms",
                "Final BFGS ms",
            ],
            rows,
        ),
        "",
    ]
    series = json.loads((D / "series.json").read_text())
    points = []
    rows = []
    for i in range(31):
        label = f"checkpoint-{i:02d}"
        speedup = math.exp(
            sum(math.log(r["checkpoint-00"] / r[label]) for r in series["summary"])
            / len(series["summary"])
        )
        br = [
            r
            for r in series["records"]
            if r["run"]
            and r["build"] == label
            and r["family"] == "orbit"
            and r["size"] == 40
        ]
        bfgs = statistics.median(r["bfgs_seconds"] for r in br) * 1000
        title = (
            "Frozen baseline"
            if not i
            else (a.root / f"iteration-{i:02d}-plan.md")
            .read_text()
            .splitlines()[0]
            .split(": ", 1)[1]
        )
        points.append(dict(iteration=i, speedup=speedup, orbit40_bfgs_ms=bfgs))
        rows.append([i, title, f"{speedup:.3f}", f"{bfgs:.3f}"])
    sections += [
        "## All checkpoint measurements",
        "",
        "Each point was remeasured in one shared, rotating, pinned three-repeat campaign. Speedup is the geometric mean of four fixture ratios to checkpoint 00. It is not a product of noisy adjacent speedups. Checkpoints use version 0.14; the final 0.15 release is measured separately above.",
        "",
        table(["Iteration", "Plan", "SQP geometric speedup", "Orbit-40 BFGS ms"], rows),
        "",
    ]
    with (D / "progress.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(points)
    (a.root / "TABLES.md").write_text(
        "# Recomputed performance tables\n\nBaseline is original main 4d776bb, final is 0.15.0. All ratios are baseline/final; larger is faster. No failed solve is counted as a speedup. Single host, repeated fresh processes; raw observations and hashes are adjacent.\n\n"
        + "\n".join(sections)
    )
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax, bx) = plt.subplots(
            2, 1, figsize=(10, 6.5), sharex=True, layout="constrained"
        )
        x = [r["iteration"] for r in points]
        ax.plot(
            x,
            [r["speedup"] for r in points],
            color="#087e8b",
            marker="o",
            markersize=3,
            linewidth=1.8,
        )
        ax.axhline(1, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Four-fixture SQP speedup")
        ax.set_title(
            "SQP-craft: 30 measured performance checkpoints",
            loc="left",
            fontweight="bold",
        )
        ax.grid(alpha=0.18)
        bx.plot(
            x,
            [r["orbit40_bfgs_ms"] for r in points],
            color="#c16b20",
            marker="o",
            markersize=3,
            linewidth=1.8,
        )
        bx.set_ylabel("Orbit-40 BFGS phase (ms)")
        bx.set_xlabel("Checkpoint (0 = frozen baseline)")
        bx.set_xticks(range(0, 31, 5))
        bx.grid(alpha=0.18)
        fig.suptitle(
            "Fixed callbacks and tolerances · one pinned numerical thread · medians of three samples",
            fontsize=10,
            color="#555555",
        )
        assets = a.root / "assets"
        assets.mkdir(exist_ok=True)
        fig.savefig(assets / "progress.png", dpi=170)
        fig.savefig(assets / "progress.svg")
        svg = assets / "progress.svg"
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
        )
        plt.close(fig)
    print("tables and checkpoint data regenerated")


if __name__ == "__main__":
    main()
