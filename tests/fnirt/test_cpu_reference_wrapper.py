"""Run the actual common subprocess timer through the frozen-tool wrapper."""

import importlib.util
import json
import os
from pathlib import Path
import sys


def _load(name, filename):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, root / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_stage_clock_preserves_common_global_usage(tmp_path):
    harness = _load("common_stage_clock_test", "benchmark_multimodal_cpu.py")
    wrapper = _load("native_stage_clock_test", "benchmark_fnirt_cpu_reference_adapter.py")
    harness.run_process = wrapper._stage_clock_wrapper(harness)
    affinity = [min(os.sched_getaffinity(0))]
    destination = tmp_path / "command_0"
    elapsed = harness.run_process([sys.executable, "-c", "pass"], destination,
                                  dict(os.environ), affinity)
    clock = json.loads((destination / "timing.private.json").read_text())
    assert clock["elapsed_seconds"] == elapsed
    assert {key: clock[key] for key in harness.run_process.last_usage} == harness.run_process.last_usage
    assert elapsed > 0
    assert clock["user_seconds"] >= 0 and clock["system_seconds"] >= 0
    candidate = tmp_path / "process"
    harness.run_process([sys.executable, "-c", "pass"], candidate, dict(os.environ), affinity)
    assert not (candidate / "timing.private.json").exists()
