"""Regenerate exact final tables, CSV and optional exported scaling figure."""

import argparse
import csv
import json
from pathlib import Path


def read_json(path):
    """`path`, or its gzip-compressed copy `path.gz`: the repository keeps the
    large raw files compressed, and fresh runs write plain JSON."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    import gzip

    with gzip.open(str(path) + ".gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


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
    p.add_argument("--root", type=Path, default=Path("docs/performance-60"))
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    d = a.root / "data"
    names = [
        "final-matrix",
        "common-baseline",
        "final-heldout",
        "final-qp",
        "final-qp-heldout",
        "final-propagation-confirmation",
        "final-mixed32",
        "baseline-confirmation",
    ]
    datasets = {name: read_json(d / (name + ".json")) for name in names}
    for name, data in datasets.items():
        assert data["status"] == "passed", name
        assert len(data["records"]) == (data["repeats"] + 1) * len(data["cases"]) * len(
            data["binaries"]
        ), name
        assert all(
            row["passed"] == row["samples"] == data["repeats"]
            for row in data["summary"]
        ), name
        assert all(
            row["returncode"] == 0 and not row["error"] for row in data["records"]
        ), name

    def find(name, family, size, backend, inner=1, outer=1, jobs=1, build="final"):
        return next(
            r
            for r in datasets[name]["summary"]
            if r["build"] == build
            and r["case"] == [family, size, backend, inner, outer, jobs]
        )

    sections = [
        "# Recomputed final performance tables",
        "",
        "Every reported sample passed the unchanged original-model assertions and driver certificate checks. Wall means setup-inclusive workflow time; native solver/QP work and process CPU time are recorded separately. Ratios are baseline/time, larger is faster. These are medians on one shared host, not monotone per-iteration guarantees.",
        "",
    ]
    for name in names:
        data = datasets[name]
        rows = []
        for r in data["summary"]:
            f, n, b, inner, outer, jobs = r["case"]
            rows.append(
                [
                    r["build"],
                    f,
                    n,
                    b,
                    inner,
                    outer,
                    jobs,
                    f"{r['seconds']*1000:.3f}",
                    f"{r['solver_seconds_per_job']*1000:.3f}",
                    "–" if r["cpu_seconds"] is None else f"{r['cpu_seconds']*1000:.3f}",
                    r["samples"],
                ]
            )
        sections += [
            "## " + name,
            "",
            table(
                [
                    "Build",
                    "Family",
                    "Size",
                    "Backend",
                    "Inner",
                    "Outer",
                    "Jobs",
                    "Workflow ms",
                    "Median native job ms",
                    "Process CPU ms",
                    "Repeats",
                ],
                rows,
            ),
            "",
        ]
    rows = []
    for c in datasets["common-baseline"]["cases"]:
        b = find("common-baseline", *c, build="baseline")
        f = find("common-baseline", *c)
        rows.append(
            [
                *c,
                f"{b['seconds']*1000:.3f}",
                f"{f['seconds']*1000:.3f}",
                f"{b['seconds']/f['seconds']:.3f}",
            ]
        )
    sections += [
        "## Same configuration: 0.15 / 0.16 comparison",
        "",
        table(
            [
                "Family",
                "Size",
                "Backend",
                "Inner",
                "Outer",
                "Jobs",
                "0.15 ms",
                "0.16 ms",
                "0.15/0.16",
            ],
            rows,
        ),
        "",
    ]
    (a.root / "TABLES.md").write_text("\n".join(sections))
    with (d / "final-scaling.csv").open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "build",
                "family",
                "size",
                "backend",
                "inner",
                "outer",
                "jobs",
                "workflow_ms",
                "median_native_job_ms",
                "process_cpu_ms",
                "repeats",
            ]
        )
        for name, data in datasets.items():
            for r in data["summary"]:
                writer.writerow(
                    [
                        name,
                        r["build"],
                        *r["case"],
                        r["seconds"] * 1000,
                        r["solver_seconds_per_job"] * 1000,
                        None if r["cpu_seconds"] is None else r["cpu_seconds"] * 1000,
                        r["samples"],
                    ]
                )
    metrics = {"propagation": {}, "batch": {}, "qp": {}, "same_configuration": []}
    for backend in ["active-set", "clarabel"]:
        serial = find("final-matrix", "propagation", 32, backend)
        parallel = find("final-matrix", "propagation", 32, backend, 8)
        metrics["propagation"][backend] = dict(
            workflow_1_ms=serial["seconds"] * 1000,
            workflow_8_ms=parallel["seconds"] * 1000,
            native_1_ms=serial["solver_seconds_per_job"] * 1000,
            native_8_ms=parallel["solver_seconds_per_job"] * 1000,
            workflow_speedup=serial["seconds"] / parallel["seconds"],
            native_speedup=serial["solver_seconds_per_job"]
            / parallel["solver_seconds_per_job"],
        )
        for family, size in [("orbit", 20), ("separable", 128)]:
            serial = find("final-matrix", family, size, backend, 1, 1, 16)
            parallel = find("final-matrix", family, size, backend, 1, 8, 16)
            metrics["batch"][f"{backend}/{family}/{size}"] = dict(
                jobs=16,
                one_worker_ms=serial["seconds"] * 1000,
                eight_workers_ms=parallel["seconds"] * 1000,
                speedup=serial["seconds"] / parallel["seconds"],
            )
    for n in [128, 256, 512, 1024]:
        serial = find("final-qp", "qp", n, "active-set")
        parallel = find("final-qp", "qp", n, "active-set", 8)
        metrics["qp"][str(n)] = dict(
            three_calls_1_ms=serial["seconds"] * 1000,
            three_calls_8_ms=parallel["seconds"] * 1000,
            speedup=serial["seconds"] / parallel["seconds"],
        )
    for c in datasets["common-baseline"]["cases"]:
        b = find("common-baseline", *c, build="baseline")
        f = find("common-baseline", *c)
        metrics["same_configuration"].append(
            dict(
                case=c,
                baseline_ms=b["seconds"] * 1000,
                final_ms=f["seconds"] * 1000,
                speedup=b["seconds"] / f["seconds"],
            )
        )
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        widths = [1, 2, 4, 8]
        colors = ["#087e8b", "#c16b20"]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
        for backend, color in zip(["active-set", "clarabel"], colors):
            values = [
                find("final-matrix", "propagation", 32, backend, w)[
                    "solver_seconds_per_job"
                ]
                * 1000
                for w in widths
            ]
            axes[0].plot(widths, values, "o-", label=backend, color=color)
        axes[0].set(
            title="Nonlinear propagation: 32 segments",
            ylabel="Native solve (ms)",
            xlabel="Inner worker budget",
        )
        for n, color in zip([512, 1024], colors):
            baseline = find("final-qp", "qp", n, "active-set")["seconds"]
            values = [
                baseline / find("final-qp", "qp", n, "active-set", w)["seconds"]
                for w in widths
            ]
            axes[1].plot(widths, values, "o-", label=f"{n} variables", color=color)
        axes[1].set(
            title="Standalone coupled QPs",
            ylabel="Speedup (three fresh-H calls)",
            xlabel="Inner worker budget",
        )
        for backend, color in zip(["active-set", "clarabel"], colors):
            baseline = find("final-matrix", "orbit", 20, backend, 1, 1, 16)["seconds"]
            values = [
                baseline
                / find("final-matrix", "orbit", 20, backend, 1, w, 16)["seconds"]
                for w in widths
            ]
            axes[2].plot(widths, values, "o-", label=backend, color=color)
        axes[2].set(
            title="Orbit20: 16 independent solves",
            ylabel="Throughput speedup",
            xlabel="Outer workers (inner 1)",
        )
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(widths, labels=list(map(str, widths)))
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
        fig.suptitle(
            "SQP-craft 0.16: measured eight-CPU scaling\nOne warmup; medians of 3 NLP / 7 QP samples; shared EPYC host; original KKT/reference checks",
            fontsize=12,
        )
        assets = a.root / "assets"
        assets.mkdir(exist_ok=True)
        fig.savefig(assets / "scaling.png", dpi=170)
        fig.savefig(assets / "scaling.svg")
        plt.close(fig)
        svg = assets / "scaling.svg"
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
        )
    print(
        "final tables, CSV, metrics"
        + (" and figure" if a.plot else "")
        + " regenerated"
    )


if __name__ == "__main__":
    main()
