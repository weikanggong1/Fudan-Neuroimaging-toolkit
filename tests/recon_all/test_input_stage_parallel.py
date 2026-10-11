"""真实计算在独立 benchmark；这里检查预算、文件保护、join与失败恢复。"""
import json
from pathlib import Path

import pytest
import torch

from fnit.recon_all import input_stage_parallel as group


@pytest.fixture
def arguments(tmp_path):
    resources = tmp_path / "resources"
    resources.mkdir()
    for name in ("orig.mgz", "LUT.txt", "synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy",
                 "synthseg_segmentation_names_2.0.npy", "synthseg_topological_classes_2.0.npy"):
        (resources / name).write_bytes(name.encode())
    return dict(orig_path=resources / "orig.mgz", n4_output=tmp_path / "out/nu0.mgz",
                segmentation_output=tmp_path / "out/synthseg.mgz", volumes_output=tmp_path / "out/vol.csv",
                weights_dir=resources, color_lut=resources / "LUT.txt", report_path=tmp_path / "out/group.json",
                device="cuda:1", threads=4)


@pytest.mark.parametrize("changes", [{"device": "cpu"}, {"device": "cuda"}, {"threads": True},
    {"threads": 4.0}, {"threads": 3}, {"execution": "threads"}])
def test_invalid_policy_before_outputs(arguments, changes):
    with pytest.raises(ValueError):
        group.run_input_stage_group(**(arguments | changes))
    assert not arguments["report_path"].parent.exists()


def test_does_not_overwrite_image_or_existing_output(arguments):
    source = arguments["orig_path"].read_bytes()
    with pytest.raises(ValueError):
        group.run_input_stage_group(**(arguments | {"n4_output": arguments["orig_path"]}))
    assert arguments["orig_path"].read_bytes() == source
    arguments["n4_output"].parent.mkdir()
    arguments["n4_output"].write_text("other person's output")
    with pytest.raises(FileExistsError):
        group.run_input_stage_group(**arguments)
    assert arguments["n4_output"].read_text() == "other person's output"


def test_missing_resource_before_creating_output(arguments):
    (arguments["weights_dir"] / "synthseg_2.0.h5").unlink()
    with pytest.raises(FileNotFoundError):
        group.run_input_stage_group(**arguments)
    assert not arguments["report_path"].parent.exists()


def install_workers(monkeypatch, *, failure=None):
    from fnit.recon_all import hemisphere_parallel, profiling
    calls, cancelled = [], []
    class Sampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def add_worker(self, pid):
            pass
        def sample_if_due(self, **kwargs):
            pass
        def report(self):
            return {"status": "ownership_unresolved", "peak_tree_total_bytes": None}
    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 12345 + len(calls)
            self.request = json.loads(Path(command[-2]).read_text())
            self.returncode = 1 if self.request["operation"] == failure else 0
            calls.append((command, kwargs, self))
            value = {}
            if self.returncode == 0:
                fields = self.request["kwargs"]
                for name in ("output_path", "segmentation_path", "volumes_path"):
                    if name in fields:
                        Path(fields[name]).write_bytes(name.encode())
                value = {"complete_operation": self.request["operation"]}
            Path(command[-1]).write_text(json.dumps({"status": "failed" if self.returncode else "complete",
                "error": "owned worker failure" if self.returncode else None, "value": value,
                "operation_started_monotonic": float(self.pid),
                "operation_finished_monotonic": float(self.pid) + .25}))
        def wait(self):
            return self.returncode
        def poll(self):
            return self.returncode
    monkeypatch.setattr(profiling, "ProcessTreeDeviceSampler", Sampler)
    monkeypatch.setattr(group.subprocess, "Popen", Process)
    monkeypatch.setattr(hemisphere_parallel, "_cancel", lambda processes: cancelled.extend(processes))
    return calls, cancelled


@pytest.mark.parametrize("execution,budgets", [("serial", [4, 4]), ("parallel", [1, 3])])
def test_complete_join_fresh_exec_and_parent_preserved(arguments, monkeypatch, execution, budgets):
    calls, cancelled = install_workers(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
    before = group._precision()
    result = group.run_input_stage_group(**arguments, execution=execution)
    assert result["status"] == "complete" and result["parent_state_preserved"]
    assert result["device_process_tree"]["peak_tree_total_bytes"] is None
    assert [call[2].request["threads"] for call in calls] == budgets
    assert [call[2].request["device"] for call in calls] == ["cpu", "cuda:1"]
    assert all(call[1]["start_new_session"] for call in calls)
    assert all("runpy.run_module" in call[0][4] for call in calls)
    assert calls[1][2].request["kwargs"]["cudnn_deterministic"] == before["cudnn_deterministic"]
    assert result["precision_after"] == before
    assert result["parent_state_after"]["current_device"] == 0
    assert result["parent_state_after"]["allocator_environment"] == "1"
    assert len(result["workers"]) == 2 and not cancelled
    assert json.loads(arguments["report_path"].read_text())["status"] == "complete"


def test_failed_worker_preserves_report_and_cancels_only_owned(arguments, monkeypatch):
    calls, cancelled = install_workers(monkeypatch, failure="synthseg")
    with pytest.raises(RuntimeError, match="synthseg worker failed"):
        group.run_input_stage_group(**arguments, execution="parallel")
    result = json.loads(arguments["report_path"].read_text())
    assert result["status"] == "failed" and result["parent_state_preserved"]
    assert result["workers"]["synthseg"]["error"] == "owned worker failure"
    assert set(cancelled) == {call[2] for call in calls}
    assert len(cancelled) == 2
    assert arguments["report_path"].with_suffix(".synthseg.log").exists()
