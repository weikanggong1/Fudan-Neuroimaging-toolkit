"""模型复用、内存输入与未消费形变的语义回归。"""
import nibabel as nib
import numpy as np
import pytest
import torch
from fnit.recon_all import aux_seg, sclimbic
from fnit.synthmorph.models import SynthMorphNetwork

@pytest.mark.parametrize("cudnn_tf32", [False, True])
def test_memory_inference_matches_mgz_roundtrip_and_reuses_model(tmp_path, monkeypatch, cudnn_tf32):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", cudnn_tf32)
    loads = []
    class Fixed(torch.nn.Module):
        def forward(self, image):
            return torch.cat((1-image, image), dim=1)
    def load(cls, path):
        loads.append(path)
        return Fixed()
    monkeypatch.setattr(sclimbic.LimbicUNet, "from_h5", classmethod(load))
    affine = np.array([[-1,0,0,12], [0,0,1,-9], [0,-1,0,17], [0,0,0,1]], float)
    data = np.arange(8**3, dtype=np.float32).reshape((8,)*3)
    native = nib.MGHImage(data, affine)
    weight = tmp_path/"two.h5"
    weight.write_bytes(b"fixture")
    rows = ((0,"Unknown"), (6101,"Left-Dura-MCA"))
    cache, reports = {}, []
    first = aux_seg._infer_crop(data, native, np.zeros(3), weight, rows, 8, "cpu",
                               model_cache=cache, precision_report=reports)
    second = aux_seg._infer_crop(data, native, np.zeros(3), weight, rows, 8, "cpu",
                                model_cache=cache, precision_report=reports)
    assert len(loads) == 1 and len(reports) == 2
    assert next(iter(cache))[2] is cudnn_tf32
    path, out, ctab = tmp_path/"input.mgz", tmp_path/"out.mgz", tmp_path/"table"
    nib.save(native, path)
    ctab.write_text("0 Unknown 0 0 0 0\n6101 Left-Dura-MCA 0 0 0 0\n")
    sclimbic.mri_sclimbic_seg(path, out, model_path=weight, ctab_path=ctab, fov=8)
    np.testing.assert_array_equal(first, np.asarray(nib.load(out).dataobj))
    np.testing.assert_array_equal(second, first)
    assert reports[0]["input_dtype"] == "torch.float32"
    assert reports[0]["cudnn_tf32"] is cudnn_tf32
    assert not reports[0]["autocast"]["enabled"]

def test_forward_only_keeps_both_velocity_forwards_and_same_forward(monkeypatch):
    import fnit.synthmorph.models as models
    model = SynthMorphNetwork.__new__(SynthMorphNetwork)
    torch.nn.Module.__init__(model)
    model.model, model.int_steps = "deform", 5
    calls, integrals = [], []
    class Deform(torch.nn.Module):
        def forward(self, moving, fixed):
            calls.append(1)
            return (moving-fixed).expand(-1,3,-1,-1,-1).clone() * 0.01
    model.deform = Deform()
    original = models.integrate
    def integrate(velocity, steps):
        integrals.append(1)
        return original(velocity, steps)
    monkeypatch.setattr(models, "integrate", integrate)
    torch.manual_seed(1)
    moving, fixed = torch.rand(1,1,32,32,32), torch.rand(1,1,32,32,32)
    forward, reverse = model(moving, fixed)
    assert len(calls) == 2 and len(integrals) == 2 and reverse is not None
    calls.clear(); integrals.clear()
    only, absent = model(moving, fixed, compute_inverse=False)
    assert len(calls) == 2 and len(integrals) == 1 and absent is None
    torch.testing.assert_close(only, forward, atol=0, rtol=0)
