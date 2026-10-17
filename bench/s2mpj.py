"""Access to the S2MPJ collection (Gratton & Toint, BSD-3-Clause): CUTEst
problems in pure Python, pinned to one commit and kept outside the
repository.

    python -m bench.s2mpj fetch [--cache DIR]

clones https://github.com/GrattonToint/S2MPJ at COMMIT into the cache
(default: ~/.cache/sqp-craft-bench, or $SQP_CRAFT_BENCH_CACHE).
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import subprocess
import sys

REPOSITORY = "https://github.com/GrattonToint/S2MPJ.git"
COMMIT = "fea6a70048eaad28b13a08703ddbfdbf65cd9c30"  # 24 August 2026


def cache_dir(cache: str | os.PathLike | None = None) -> pathlib.Path:
    base = cache or os.environ.get("SQP_CRAFT_BENCH_CACHE") or "~/.cache/sqp-craft-bench"
    return pathlib.Path(base).expanduser()


def checkout(cache: str | os.PathLike | None = None) -> pathlib.Path:
    """The S2MPJ checkout at COMMIT, cloned on first use."""
    root = cache_dir(cache) / "S2MPJ"
    if not (root / "s2mpjlib.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", REPOSITORY, str(root)], check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(root), "fetch", "--quiet", "origin", COMMIT], check=False)
        subprocess.run(["git", "-C", str(root), "checkout", "--quiet", COMMIT], check=True)
    return root


def problem_names(cache=None) -> list[str]:
    root = checkout(cache)
    return sorted(p.stem for p in (root / "python_problems").glob("*.py"))


def load(name: str, cache=None):
    """An instance of S2MPJ problem `name` with its default parameters."""
    root = checkout(cache)
    for path in (root, root / "python_problems"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module(name)
    return getattr(module, name)()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch"])
    parser.add_argument("--cache")
    args = parser.parse_args()
    print(checkout(args.cache))


if __name__ == "__main__":
    main()
