"""False performance wins must be rejected before producing summary statistics."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def mock(tmp_path, mode):
    exe = tmp_path / "benchmark"
    marker = tmp_path / "started"
    exe.write_text(
        "#!/usr/bin/env python3\n" + f"MODE={mode!r}\nMARKER={str(marker)!r}\n" + """
import json,sys
from pathlib import Path
Path(MARKER).write_text("started")
row=dict(family="separable",size=128,seconds=0.1,objective=0.,iterations=1,primal=0.,stationarity=0.,complementarity=0.,dual_sign=0.)
if MODE=="bad":row["stationarity"]=0.01
if MODE=="nan":row["stationarity"]=float("nan")
print(json.dumps(row));print(json.dumps(dict(kind="batch",jobs=1,workers=1,seconds=0.1)))
if MODE=="fail":
    print("single-threaded solves unimplemented",file=sys.stderr)
    sys.exit(1)
"""
    )
    exe.chmod(0o755)
    return exe, marker


def run(tmp_path, mode, case="separable:128:active-set:1:1:1"):
    exe, marker = mock(tmp_path, mode)
    output = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/performance60.py"),
            str(output),
            "candidate=" + str(exe),
            "--repeats",
            "1",
            "--case",
            case,
        ],
        capture_output=True,
        text=True,
    )
    return result, output, marker


@pytest.mark.parametrize("mode", ["bad", "nan", "fail"])
def test_failed_or_invalid_benchmarks_never_become_speedups(tmp_path, mode):
    result, output, _ = run(tmp_path, mode)
    assert result.returncode != 0
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "failed" and "summary" not in evidence
    assert evidence["records"][0]["error"]


def test_invalid_budget_launches_no_child(tmp_path):
    result, output, marker = run(tmp_path, "good", "separable:128:active-set:8:8:1")
    assert result.returncode != 0 and not marker.exists() and not output.exists()


def test_complete_valid_record_keeps_warmup_separate(tmp_path):
    result, output, _ = run(tmp_path, "good")
    assert result.returncode == 0, result.stderr
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "passed" and len(evidence["records"]) == 2
    assert evidence["summary"][0]["samples"] == 1
