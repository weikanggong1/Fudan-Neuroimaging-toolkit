"""Python候选质量门接线；真实网格回归另见带哈希报告。"""
from pathlib import Path
from unittest.mock import patch

import pytest

from fnit.recon_all import native_free
from fnit.recon_all.mris_register_overlap import SphereOverlapQualityError
from fnit.recon_all.mris_register_run import run_register_sphere


@pytest.mark.parametrize("strict", [False, True])
def test_full_registration_forwards_quality_policy(tmp_path, strict):
    inputs = {}
    for name in ("sphere", "smoothwm", "sulc", "atlas"):
        inputs[name] = tmp_path / name
        inputs[name].write_bytes(name.encode())
    output = tmp_path / "sphere.reg"

    def sulc_stage(sphere, smoothwm, sulc, atlas, seed, **kwargs):
        seed.write_bytes(b"same-seed")
        return {"last_iteration": 219, "output": str(seed)}

    def smooth_stage(sphere, smoothwm, seed, atlas, target, **kwargs):
        assert kwargs["require_unfolded"] is strict
        assert kwargs["overlap_diagnostics"] is True
        assert kwargs["seed_iteration"] == 219
        target.write_bytes(b"same-geometry")
        return {"sulc_seed": str(seed), "overlap_diagnostics": {"quality_status": "passed"}}

    with patch("fnit.recon_all.mris_register_run.run_register_sulc", side_effect=sulc_stage), \
         patch("fnit.recon_all.mris_register_run.run_register_smoothwm", side_effect=smooth_stage):
        result = run_register_sphere(**inputs, output=output, require_unfolded=strict,
                                     overlap_diagnostics=True)
    assert result["require_unfolded"] is strict
    assert result["smoothwm_pass"]["overlap_diagnostics"]["quality_status"] == "passed"


def test_registration_quality_failure_preserves_diagnostics_and_removes_seed(tmp_path):
    inputs = {}
    for name in ("sphere", "smoothwm", "sulc", "atlas"):
        inputs[name] = tmp_path / name
        inputs[name].write_bytes(name.encode())
    output = tmp_path / "sphere.reg"
    quality = {"final_negative_float32": 7507, "final_negative_double": 7507,
               "stop_reason": "source_niterations_limit"}

    def sulc_stage(sphere, smoothwm, sulc, atlas, seed, **kwargs):
        seed.write_bytes(b"same-seed")
        return {"last_iteration": 219}

    def reject(*args, **kwargs):
        assert kwargs["require_unfolded"] is True
        raise SphereOverlapQualityError(quality)

    with patch("fnit.recon_all.mris_register_run.run_register_sulc", side_effect=sulc_stage), \
         patch("fnit.recon_all.mris_register_run.run_register_smoothwm", side_effect=reject):
        with pytest.raises(SphereOverlapQualityError) as error:
            run_register_sphere(**inputs, output=output, require_unfolded=True)
    assert error.value.diagnostics == quality
    assert not output.exists()
    assert not list(tmp_path.glob("fnit-sphere-reg-*"))


@pytest.mark.parametrize("strict", [False, True])
def test_standard_sphere_pipeline_forwards_quality_policy(tmp_path, strict):
    with patch("fnit.recon_all.inflate_standard_run.run_standard_inflate",
               return_value={"total_seconds_including_io": 1.0}), \
         patch("fnit.recon_all.sphere_standard_run.run_standard_sphere",
               return_value={"total_seconds_including_io": 2.0}) as run:
        timings, report = native_free._run_accurate_sphere_pair(
            inflate_binary=None, subject=tmp_path, hemi="lh", assets=tmp_path,
            device="cuda:0", inflate_backend="torch", sphere_finish_backend="torch",
            require_unfolded=strict)
    assert run.call_args.kwargs.get("require_unfolded", False) is strict
    assert timings == {"inflate": 1.0, "sphere": 2.0}


def test_worker_preserves_quality_failure_as_json(tmp_path, monkeypatch, capsys):
    import importlib
    import json
    import sys
    import types
    import torch
    from fnit.recon_all import hemisphere_worker

    quality = {"final_negative_float32": 7507, "final_negative_double": 7507,
               "stop_reason": "source_niterations_limit"}
    def reject(**kwargs):
        raise SphereOverlapQualityError(quality)

    request = tmp_path / "request.json"
    output = tmp_path / "worker.json"
    request.write_text(json.dumps({"device": "cpu", "threads": 1,
        "precision": {"matmul_tf32": True, "cudnn_tf32": False},
        "profile_stages": False, "callable": "fnit_test_quality:reject",
        "operation": "sphere", "kwargs": {}}))
    actual_import = importlib.import_module
    def selected_import(name, *args, **kwargs):
        if name == "fnit_test_quality":
            return types.SimpleNamespace(reject=reject)
        return actual_import(name, *args, **kwargs)
    monkeypatch.setattr(sys, "argv", ["worker", str(request), str(output)])
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(importlib, "import_module", selected_import)
    with pytest.raises(SystemExit) as error:
        hemisphere_worker.main()
    assert error.value.code == 1
    result = json.loads(output.read_text())
    assert result["quality_failure"] == quality
    assert result["status"] == "failed" and not result["cuda_oom"]
