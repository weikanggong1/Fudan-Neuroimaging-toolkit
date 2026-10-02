"""验证仿射几何的局部matmul精度，模拟模型仅用于调用语义，不代替真实benchmark。"""
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import pytest
import torch
from fnit.recon_all import mni_aux_chain


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("initial_matmul_tf32", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_affine_precision_scope_and_restore(tmp_path, monkeypatch, device, initial_matmul_tf32, fail):
    subject, weights, assets = (tmp_path/name for name in ("subject", "weights", "assets"))
    (subject/"mri").mkdir(parents=True)
    weights.mkdir()
    target = assets/mni_aux_chain.TEMPLATE_DIR
    target.mkdir(parents=True)
    values = np.ones((8,)*3, np.float32)
    nib.save(nib.MGHImage(values, np.eye(4)), subject/"mri/orig.mgz")
    for name in ("mni152.1.0mm.cropped.nii.gz", "mni152.1.0mm.nii.gz"):
        nib.save(nib.Nifti1Image(values, np.eye(4)), target/name)
    (weights/"synthmorph.affine.2.h5").write_bytes(b"model fixture")
    seen = []

    class World:
        matrix = np.eye(4)
        def save(self, path):
            from pathlib import Path
            Path(path).write_text("world affine fixture\n")

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["configure_precision"] is False
            assert kwargs["device"] == device
        def __call__(self, *args, **kwargs):
            seen.append(torch.backends.cuda.matmul.allow_tf32)
            assert torch.backends.cudnn.allow_tf32
            if fail:
                raise RuntimeError("affine failed")
            return SimpleNamespace(transform=World())

    monkeypatch.setattr(mni_aux_chain, "SynthMorph", Model)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", initial_matmul_tf32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    arguments = dict(subject_dir=subject, weights_dir=weights, assets_dir=assets,
                     device=device, threads=4, precision_report=[])
    if fail:
        with pytest.raises(RuntimeError, match="affine failed"):
            mni_aux_chain.register_mni152_affine(**arguments)
    else:
        output = mni_aux_chain.register_mni152_affine(**arguments)
        assert output.is_file()
    assert seen == [False if device.startswith("cuda") else initial_matmul_tf32]
    assert torch.backends.cuda.matmul.allow_tf32 is initial_matmul_tf32
    assert torch.backends.cudnn.allow_tf32
