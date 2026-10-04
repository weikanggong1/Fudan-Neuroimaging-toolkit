"""CPU recon entry preserves an unrelated caller's CUDA precision policy."""
import json
from pathlib import Path

import pytest
import torch

from fnit.recon_all import assets, input_talairach_chain, mni_aux_chain
from fnit.recon_all import native_free, native_runtime_selection
from fnit import weights


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("initial", [False, True])
def test_recon_device_policy_at_real_first_stage(tmp_path, monkeypatch, device, initial):
    # Resource resolution is stubbed so this test exercises the public wrapper,
    # report construction and first-stage failure without installing C++ tools.
    t1 = tmp_path / "input.nii.gz"
    t1.write_bytes(b"resource fixture")
    directories = [tmp_path / name for name in ("weights", "assets", "bin")]
    for directory in directories:
        directory.mkdir()
    monkeypatch.setattr(mni_aux_chain, "validate_mni_aux_assets", lambda *a: None)
    monkeypatch.setattr(assets, "validate_core_assets", lambda *a: None)
    monkeypatch.setattr(weights, "verify_file", lambda *a: True)
    monkeypatch.setattr(native_free, "_native_binary", lambda folder, name: (Path(folder) / name, "fixture"))
    monkeypatch.setattr(native_free, "_folding_atlas", lambda folder, hemi: Path(folder) / hemi)
    monkeypatch.setattr(native_runtime_selection, "select_native_optimizations",
                        lambda folder, **k: {"white_binary": "mris_place_surface", "em_backend": "original"})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", initial)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial)
    expected = True if device.startswith("cuda") else initial

    def first_stage(*args, **kwargs):
        assert torch.backends.cuda.matmul.allow_tf32 is expected
        assert torch.backends.cudnn.allow_tf32 is expected
        raise RuntimeError("deliberate first-stage failure")

    monkeypatch.setattr(input_talairach_chain, "run_input_talairach_chain", first_stage)
    subject = tmp_path / "subject"
    with pytest.raises(RuntimeError, match="deliberate first-stage failure"):
        native_free.run_recon_all_python(t1, subject, directories[0], directories[1],
                                        native_bin_dir=directories[2], device=device, threads=1)
    assert torch.backends.cuda.matmul.allow_tf32 is expected
    assert torch.backends.cudnn.allow_tf32 is expected
    report = json.loads((subject / "fnit-native-free-run.json").read_text())
    assert report["precision"]["cuda_policy_applied"] is device.startswith("cuda")
    assert report["failed_stage"] == "input_talairach"
