"""监测入口必须复用共享归属器，未知宿主PID和空采样不能写为零。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("unknown,empty", [(True, False), (False, False), (False, True)])
def test_shared_sampler_uses_explicit_uuid_and_preserves_unknown(tmp_path, monkeypatch, unknown, empty):
    path = Path(__file__).parents[2] / "validation/recon_all/python_gpu_port/monitor_placement_benchmark.py"
    spec = importlib.util.spec_from_file_location("placement_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = []

    class Process:
        pid = 314
        calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 and not empty else 0

        def wait(self):
            return 0

    class Sampler:
        def __init__(self, **kwargs):
            observed.append(kwargs)
            self.uuid = None

        def sample_if_due(self, *, force):
            assert force and self.uuid == "GPU-explicit-physical-2"

        def report(self):
            samples = [] if empty else [{"monotonic": module.time.monotonic(),
                "target_device_used_bytes": 7 * 1048576,
                "target_compute_process_sum_bytes": 6 * 1048576,
                "tree_total_bytes": None if unknown else 3 * 1048576,
                "ownership": "unresolved" if unknown else "resolved", "processes": [],
                "sample_query_seconds": .001}]
            return {"samples": samples, "failed_samples": []}

    monkeypatch.setattr(module, "load_sampler_module", lambda _: SimpleNamespace(
        ProcessTreeDeviceSampler=Sampler, __file__=str(path)))
    monkeypatch.setattr(module, "query", lambda _: [["2", "GPU-explicit-physical-2", "A100"]])
    monkeypatch.setattr(module.subprocess, "Popen", lambda _: Process())
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    output = tmp_path / "memory.json"
    monkeypatch.setattr("sys.argv", [str(path), "--physical-gpu", "2", "--output", str(output),
                                    "--interval-seconds", ".25", "--", "/usr/bin/true"])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    assert observed == [{"device": "cuda:0", "parent_pid": 314, "interval": .25}]
    report = json.loads(output.read_text())
    assert report["sampled_peak_benchmark_tree_used_mib"] == (None if unknown or empty else 3)
    assert report["sampled_peak_device_used_mib"] == (None if empty else 7)
    assert report["process_tree_memory_status"].startswith(
        "unavailable" if empty else "incomplete" if unknown else "mapped")
