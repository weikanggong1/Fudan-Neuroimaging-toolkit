"""Contracts for benchmark observation, not synthetic performance evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


DRIVER = Path(__file__).resolve().parents[2] / "validation/dmri_pipeline/public10_20261002/benchmark_fnit.py"
spec = importlib.util.spec_from_file_location("public10_fnit_benchmark", DRIVER)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class FakeCuda:
    def __init__(self):
        self.allocated = 0
        self.reserved = 0
        self.current = 0
        self.reset_calls = 0

    def synchronize(self, device):
        pass

    def current_device(self):
        return 0

    def max_memory_allocated(self, device):
        return self.allocated

    def max_memory_reserved(self, device):
        return self.reserved

    def memory_allocated(self, device):
        return self.current

    def reset_peak_memory_stats(self, device=None):
        self.reset_calls += 1
        self.allocated = self.current
        self.reserved = self.current


def fake_torch():
    return SimpleNamespace(cuda=FakeCuda(), device=lambda value: SimpleNamespace(index=0))


def test_capture_peak_before_nested_component_resets_and_restore_functions():
    torch = fake_torch()
    original_reset = torch.cuda.reset_peak_memory_stats
    original_child = None

    def child():
        torch.cuda.allocated = 900
        torch.cuda.reserved = 1100
        torch.cuda.current = 10
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.allocated = 200
        torch.cuda.reserved = 300
        return "actual result"

    def parent():
        return functions.child()

    functions = SimpleNamespace(child=child, parent=parent)
    original_child = functions.child
    with benchmark.StageRecorder(torch, SimpleNamespace(index=0), emit=False) as recorder:
        recorder.patch(functions, "parent", "parent")
        recorder.patch(functions, "child", "child")
        assert functions.parent() == "actual result"
    assert recorder.peak_allocated == 900
    assert recorder.peak_reserved == 1100
    assert recorder.checkpoints[0]["active_stages"] == ["parent", "child"]
    assert recorder.events[0]["parent_stage"] == "parent"
    assert recorder.events[1]["parent_stage"] is None
    assert recorder.events[1]["peak_allocated_bytes"] == 900
    assert torch.cuda.reset_calls == 1  # no extra resets added to the production call
    assert torch.cuda.reset_peak_memory_stats == original_reset
    assert functions.child is original_child


def test_failure_keeps_stage_record_and_restores_functions():
    torch = fake_torch()

    def fail():
        raise ValueError("actual failure")

    functions = SimpleNamespace(call=fail)
    with pytest.raises(ValueError, match="actual failure"):
        with benchmark.StageRecorder(torch, SimpleNamespace(index=0), emit=False) as recorder:
            recorder.patch(functions, "call", "failing stage")
            functions.call()
    assert recorder.events[0]["status"] == "failed"
    assert functions.call is fail


def test_private_paths_are_removed_but_source_operation_text_is_retained():
    qc = {"path": "/private/sub/person/file.nii.gz", "windows": "C:\\private\\file.nii.gz",
          "source": "preprocessing/inference/native-grid", "nested": [Path("/private/file")]}
    public = benchmark.public_value(qc)
    assert public["path"] == "[private filesystem path]"
    assert public["windows"] == "[private filesystem path]"
    assert public["nested"] == ["[private filesystem path]"]
    assert public["source"] == qc["source"]


def test_source_hash_streams_exact_bytes(tmp_path):
    import hashlib
    resource = tmp_path / "weights.pt"
    contents = b"fixed bytes" * 20
    resource.write_bytes(contents)
    assert benchmark.file_provenance(resource) == {
        "size_bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
