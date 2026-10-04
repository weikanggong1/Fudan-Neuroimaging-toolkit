"""CUDA entry points enable TF32 while retaining float32 tensors."""

import pytest
import torch

from fnit.bedpostx import TorchBEDPOSTX
from fnit.probtrackx import TorchProbtrackX
from fnit.applywarp import TorchApplyWarp
from fnit.fast import TorchFAST
from fnit.fast_vbm import FastVBM
from fnit.mmorf import TorchMMORF
from fnit.dmri_pipeline import DMRIPipeline
from fnit.synthstrip import SynthStrip
from fnit.synthmorph import SynthMorph
from fnit.synthsr import SynthSR
from fnit.flirt import TorchFLIRT
from fnit.fnirt import TorchFNIRT
from fnit.topup import TorchTOPUP
from fnit.eddy import TorchEDDY
from fnit.dtifit import TorchDTIFIT
from fnit.amico_noddi import TorchAMICONODDI
from fnit.wmh_synthseg import WMHSynthSeg
from fnit.synthseg_parc import SynthSeg, SynthSegParc, SynthSegSegmenter


@pytest.mark.parametrize(
    "constructor",
    (TorchApplyWarp, TorchFAST, TorchFLIRT, TorchFNIRT, TorchTOPUP,
     TorchEDDY, TorchDTIFIT, TorchAMICONODDI,
     TorchBEDPOSTX, TorchProbtrackX),
)
def test_cuda_registration_components_enable_tf32(monkeypatch, constructor):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    constructor(device="cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


@pytest.mark.parametrize(
    "constructor",
    (FastVBM, TorchMMORF, DMRIPipeline),
)
def test_cuda_composite_components_enable_tf32(monkeypatch, constructor):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    constructor(device="cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_cuda_synthstrip_enables_tf32_before_weight_resolution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    with pytest.raises(FileNotFoundError):
        SynthStrip(weights="missing-weights", device="cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_cuda_synthmorph_enables_tf32(monkeypatch):
    from fnit.synthmorph import models, pipeline

    class DummyNetwork:
        def __init__(self, **kwargs):
            self.arguments = kwargs

    monkeypatch.setattr(pipeline, "resolve_weights", lambda *args, **kwargs: "weights")
    monkeypatch.setattr(models, "SynthMorphNetwork", DummyNetwork)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    SynthMorph(weights="weights", device="cuda:0", model="deform")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_cuda_synthsr_enables_tf32(monkeypatch):
    from fnit.synthsr import pipeline

    class DummyNetwork:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(pipeline, "resolve_weights", lambda *args, **kwargs: "weights")
    monkeypatch.setattr(pipeline, "SynthSRUNet", DummyNetwork)
    monkeypatch.setattr(pipeline, "load_h5_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    SynthSR(weights="weights", device="cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


@pytest.mark.parametrize(
    ("constructor", "arguments"),
    (
        (WMHSynthSeg, ("missing-weights",)),
        (SynthSegParc, ("missing-weights", "missing-labels")),
    ),
)
def test_cuda_segmentation_components_enable_tf32(
    monkeypatch, constructor, arguments
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    with pytest.raises(FileNotFoundError):
        constructor(*arguments, device="cuda:0")

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("constructor,arguments", [
    (SynthSeg, ("missing-weights",)),
    (SynthSegSegmenter, ("missing-weights", "missing-labels")),
])
def test_scoped_segmentation_construction_preserves_cuda_policy(
        monkeypatch, initial, constructor, arguments):
    # These two APIs apply CUDA precision only while running posterior(),
    # and retain the caller's policy even when resource resolution fails.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", initial)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial)
    with pytest.raises(FileNotFoundError):
        constructor(*arguments, device="cuda:0")
    assert torch.backends.cuda.matmul.allow_tf32 is initial
    assert torch.backends.cudnn.allow_tf32 is initial
