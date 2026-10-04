"""阶段之间缺包时使用实际公开入口和线程作用域，保留部分阶段证据。"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numba
import pytest
import torch

from fnit.recon_all import native_free


def test_between_stage_import_failure_finalizes_partial_report_and_restores_threads(
        monkeypatch, tmp_path):
    subject = tmp_path / "subject"
    report_path = subject / "fnit-native-free-run.json"
    original = ModuleNotFoundError("No module named 'tifffile'", name="tifffile")
    # Numba 首次查询初始化其线程池；在进入被测作用域前先完成初始化。
    before_numba = numba.get_num_threads()
    before_torch = torch.get_num_threads()
    requested = 2 if before_torch == before_numba == 1 and numba.config.NUMBA_NUM_THREADS >= 2 else 1
    completed_stage = {"name": "sphere_lh", "status": "complete", "seconds": 4.0}

    def fail_between_stages(**kwargs):
        assert torch.get_num_threads() == requested
        assert numba.get_num_threads() == requested
        subject.mkdir()
        # 成功阶段已保存，但下一阶段的导入发生在 stage() 错误捕获之外。
        report_path.write_text(json.dumps({
            "status": "running", "stages": [completed_stage],
            "timing": {"validation_seconds": 0.25},
        }))
        raise original

    monkeypatch.setattr(native_free, "_run_recon_all_python", fail_between_stages)
    monkeypatch.setattr(native_free, "time", SimpleNamespace(
        perf_counter=Mock(side_effect=[100.0, 110.0])))

    with pytest.raises(ModuleNotFoundError) as raised:
        native_free.run_recon_all_python(
            t1="raw.nii.gz", subject_dir=subject, weights_dir="weights",
            assets_dir="assets", device="cpu", threads=requested)

    assert raised.value is original
    assert raised.value.name == "tifffile"
    assert torch.get_num_threads() == before_torch
    assert numba.get_num_threads() == before_numba
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failed_stage"] == "pipeline"
    assert report["error"] == repr(original)
    assert report["stages"] == [completed_stage]
    assert report["total_seconds"] == 10.0
    assert report["timing"]["thread_setup_and_restore_seconds"] is None
    assert "internal total_seconds was not finalized" in report["timing"]["thread_setup_and_restore_scope"]
    assert report["thread_budget"]["restoration_complete"] is True
    assert report["thread_budget"]["torch"]["restored"] == before_torch
    assert report["thread_budget"]["numba"]["restored"] == before_numba
    assert not any("KeyError" in note for note in getattr(original, "__notes__", []))


def test_missing_tifffile_is_checked_before_pipeline_imports_and_outputs(
        monkeypatch, tmp_path):
    original = ModuleNotFoundError("No module named 'tifffile'", name="tifffile")
    preflight = Mock(side_effect=original)
    monkeypatch.setattr(native_free, "_validate_standard_python_dependencies", preflight)
    t1, weights, assets = tmp_path / "raw.nii.gz", tmp_path / "weights", tmp_path / "assets"
    t1.touch()
    weights.mkdir()
    assets.mkdir()
    subject = tmp_path / "subject"
    with pytest.raises(ModuleNotFoundError) as raised:
        native_free._run_recon_all_python(
            t1=t1, subject_dir=subject, weights_dir=weights,
            assets_dir=assets, device="cpu", threads=1)
    assert raised.value is original
    preflight.assert_called_once_with()
    assert not subject.exists()


def test_preflight_imports_complete_registration_module_without_running_it(monkeypatch):
    registration = Mock()
    importer = Mock(side_effect=[SimpleNamespace(), SimpleNamespace(
        run_register_sphere=registration)])
    monkeypatch.setattr(native_free.importlib, "import_module", importer)
    native_free._validate_standard_python_dependencies()
    assert [call.args[0] for call in importer.call_args_list] == [
        "tifffile", "fnit.recon_all.mris_register_run"]
    registration.assert_not_called()


def test_invalid_input_preserves_validation_error_before_dependency_preflight(
        monkeypatch, tmp_path):
    preflight = Mock(side_effect=ModuleNotFoundError("No module named 'tifffile'"))
    monkeypatch.setattr(native_free, "_validate_standard_python_dependencies", preflight)
    with pytest.raises(FileNotFoundError, match="T1, weights, and assets must exist"):
        native_free._run_recon_all_python(
            t1=tmp_path / "missing.nii.gz", subject_dir=tmp_path / "subject",
            weights_dir=tmp_path / "weights", assets_dir=tmp_path / "assets",
            device="cpu", threads=1)
    preflight.assert_not_called()
