"""GPU 设备选择、可选同步及已初始化 API allocator 的行为回归。"""

from unittest.mock import Mock

import pytest
import torch

from fnit.recon_all.profiling import StageProfiler, configure_cuda_allocator


def test_production_does_not_synchronize_or_reset(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    sync = Mock()
    reset = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    profiler = StageProfiler(device="cuda:1", synchronize=False,
                             allocator={"torch_stats_known_unavailable": True})
    assert profiler.run("cpu-stage", lambda: 7) == 7
    sync.assert_not_called()
    reset.assert_not_called()
    assert profiler.last_row["torch_memory_stats_status"] == "unavailable"
    assert "gpu_peak_allocated_bytes" not in profiler.last_row


def test_profile_targets_requested_device_and_records_function_exception(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    allocated = Mock(return_value=100)
    reserved = Mock(return_value=200)
    reset = Mock()
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", reserved)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    profiler = StageProfiler(device="cuda:1", synchronize=True,
                             allocator={"torch_stats_known_unavailable": False,
                                        "torch_stats_known_valid": True})
    assert profiler.run("ok", lambda: 1) == 1
    assert all(call.args == (torch.device("cuda:1"),) for call in sync.call_args_list)
    assert all(call.args == (torch.device("cuda:1"),) for call in allocated.call_args_list)
    assert profiler.last_row["gpu_peak_reserved_bytes"] == 200
    assert profiler.last_row["cuda_synchronized"]
    assert profiler.last_row["seconds"] >= profiler.last_row["function_seconds"]
    with pytest.raises(RuntimeError, match="original-error"):
        profiler.run("fail", lambda: (_ for _ in ()).throw(RuntimeError("original-error")))
    assert profiler.last_row["name"] == "fail"
    assert "original-error" in profiler.last_row["error"]


def test_initialized_allocator_is_preserved_and_not_inferred_from_late_environment(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setenv("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
    report = configure_cuda_allocator("cuda:1", "auto")
    assert report["effective"] == "preserved_preinitialized_unknown"
    assert not report["torch_stats_known_unavailable"]
    with pytest.raises(ValueError, match="before CUDA initialization"):
        configure_cuda_allocator("cuda:1", "enabled")


def test_allocator_selection_before_context(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("PYTORCH_NO_CUDA_MEMORY_CACHING", raising=False)
    assert configure_cuda_allocator("cuda:1")["effective"] == "disabled"
    assert configure_cuda_allocator("cuda:1", "enabled")["effective"] == "enabled"


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_existing_allocator_environment_is_presence_based(monkeypatch, value):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("PYTORCH_NO_CUDA_MEMORY_CACHING", value)
    report = configure_cuda_allocator("cuda:1", "auto")
    assert report["effective"] == "disabled"
    assert report["torch_stats_known_unavailable"]
    assert not report["torch_stats_known_valid"]
    assert report["environment_after_selection"] == value


def test_failed_presynchronization_records_current_stage_and_preserves_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=RuntimeError("sync-failure")))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", Mock(side_effect=RuntimeError("stats-failure")))
    body = Mock()
    profiler = StageProfiler(device="cuda:1", synchronize=True,
        allocator={"torch_stats_known_unavailable": False, "torch_stats_known_valid": True})
    profiler.last_row = {"name": "previous"}
    with pytest.raises(RuntimeError, match="sync-failure"):
        profiler.run("current", body)
    body.assert_not_called()
    assert profiler.last_row["name"] == "current"
    assert "sync-failure" in profiler.last_row["error"]
    assert profiler.last_row["function_seconds"] == 0
    assert profiler.last_row["torch_memory_stats_status"] == "failed"
