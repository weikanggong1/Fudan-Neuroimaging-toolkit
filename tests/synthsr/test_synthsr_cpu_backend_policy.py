"""A CPU SynthSR instance preserves a caller's GPU backend settings."""

import torch

from fnit.synthsr import pipeline


def test_cpu_constructor_preserves_cuda_policy(monkeypatch, tmp_path):
    checkpoint = tmp_path / "unloaded_synthsr.h5"
    monkeypatch.setattr(pipeline, "resolve_weights", lambda name, explicit: checkpoint)
    monkeypatch.setattr(pipeline, "SynthSRUNet", torch.nn.Module)
    monkeypatch.setattr(pipeline, "load_h5_weights", lambda model, path: None)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    model = pipeline.SynthSR(weights=checkpoint, device="cpu")

    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert model.model.training is False
