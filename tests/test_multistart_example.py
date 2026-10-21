"""examples/multistart_processes.py counts the CPUs it may use: the affinity
mask lowered to the cgroup CPU quota (v2 and v1)."""
import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).parents[1] / "examples" / "multistart_processes.py"
    spec = importlib.util.spec_from_file_location("multistart_processes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cgroup_quotas(tmp_path):
    m = load()
    v2 = tmp_path / "v2"
    (v2 / "a" / "b").mkdir(parents=True)
    (v2 / "a" / "cpu.max").write_text("800000 100000\n")  # 8 CPUs, an ancestor
    (v2 / "a" / "b" / "cpu.max").write_text("max 100000\n")
    (tmp_path / "cg2").write_text("0::/a/b\n")
    assert m.cgroup_cpu_quota(tmp_path / "cg2", v2) == 8.0
    assert m.available_cpus(tmp_path / "cg2", v2) <= 8
    v1 = tmp_path / "v1" / "cpu,cpuacct" / "c"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("150000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    (tmp_path / "cg1").write_text("5:cpu,cpuacct:/c\n2:memory:/c\n")
    assert m.cgroup_cpu_quota(tmp_path / "cg1", tmp_path / "v1") == 1.5
    assert m.available_cpus(tmp_path / "cg1", tmp_path / "v1") == 1
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")  # no limit
    assert m.cgroup_cpu_quota(tmp_path / "cg1", tmp_path / "v1") is None
    assert m.available_cpus(tmp_path / "missing", v2) >= 1
