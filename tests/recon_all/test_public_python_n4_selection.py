"""公开Python流程保持显式N4选择；非法组合在输出/CUDA之前失败。"""
import contextlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fnit.recon_all import native_free
from fnit.recon_all.batch import run_recon_all_python_batch


@pytest.mark.parametrize("requested,execution,effective", [
    ("native", "in-process", "simpleitk"),
    ("simpleitk", "in-process", "simpleitk"),
    ("torch", "in-process", "torch"),
    ("torch", "isolated", "torch"),
])
def test_public_api_preserves_requested_n4(monkeypatch, tmp_path, requested, execution, effective):
    @contextlib.contextmanager
    def budget(**kwargs):
        yield {"requested_threads": kwargs["threads"]}
    monkeypatch.setattr("fnit.recon_all.thread_budget.thread_budget", budget)
    internal = Mock(return_value={"status": "complete", "total_seconds": 0.})
    monkeypatch.setattr(native_free, "_run_recon_all_python", internal)
    (tmp_path / "subject").mkdir()
    native_free.run_recon_all_python(t1="raw", subject_dir=tmp_path / "subject",
        weights_dir="weights", assets_dir="assets", backend="python", device="cuda:2",
        threads=4, n4_backend=requested, n4_execution=execution)
    kwargs = internal.call_args.kwargs
    assert kwargs["n4_backend"] == effective
    assert kwargs.get("n4_execution", "in-process") == execution
    assert kwargs.get("native_bin_dir") is None


@pytest.mark.parametrize("options", [
    {"n4_backend": "blur"},
    {"n4_backend": "simpleitk", "n4_execution": "isolated"},
    {"n4_backend": "native", "n4_execution": "isolated"},
    {"n4_backend": "torch", "input_execution": "parallel"},
])
def test_invalid_public_selection_is_not_silently_overridden(monkeypatch, tmp_path, options):
    budget = Mock(side_effect=AssertionError("invalid request reached thread/CUDA setup"))
    monkeypatch.setattr("fnit.recon_all.thread_budget.thread_budget", budget)
    with pytest.raises(ValueError):
        native_free.run_recon_all_python(t1="raw", subject_dir=tmp_path / "absent",
            weights_dir="weights", assets_dir="assets", backend="python", device="cuda:2",
            threads=4, **options)
    budget.assert_not_called()
    assert not (tmp_path / "absent").exists()


def test_batch_torch_isolated_selection_reaches_actual_cli(monkeypatch, tmp_path):
    raw = tmp_path / "raw.nii.gz"
    raw.write_bytes(b"routing contract, not real MRI")
    weights, assets = tmp_path / "weights", tmp_path / "assets"
    weights.mkdir(); assets.mkdir()
    subject = tmp_path / "subject"
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        subject.mkdir()
        (subject / "fnit-native-free-run.json").write_text(json.dumps({"status": "complete"}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("fnit.recon_all.batch.subprocess.run", run)
    run_recon_all_python_batch(jobs=[{"t1": raw, "subject_dir": subject}],
        weights_dir=weights, assets_dir=assets, devices=["cuda:2"], threads=4,
        backend="python", n4_backend="torch", n4_execution="isolated")
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--n4-backend") + 1] == "torch"
    assert command[command.index("--n4-execution") + 1] == "isolated"


def test_batch_rejects_parallel_torch_before_paths_or_worker(monkeypatch, tmp_path):
    child = Mock(side_effect=AssertionError("invalid request launched a child"))
    monkeypatch.setattr("fnit.recon_all.batch.subprocess.run", child)
    with pytest.raises(ValueError, match="parallel input requires simpleitk"):
        run_recon_all_python_batch(jobs=[], weights_dir=tmp_path / "missing-weights",
            assets_dir=tmp_path / "missing-assets", devices=["cuda:2"], threads=4,
            backend="python", n4_backend="torch", n4_execution="isolated",
            input_execution="parallel")
    child.assert_not_called()
