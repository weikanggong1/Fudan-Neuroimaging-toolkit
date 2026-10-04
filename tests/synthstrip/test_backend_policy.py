"""SynthStrip fixes convolution selection while retaining default TF32."""

import torch

from fnit.synthstrip import pipeline


def _prepare_constructor_without_inference(monkeypatch, tmp_path):
    # No weights, downloads, GPU, or inference are needed to verify the policy.
    checkpoint_path = tmp_path / "unloaded_synthstrip.1.pt"
    checkpoint_state = {}
    monkeypatch.setattr(pipeline, "resolve_weights", lambda name, explicit: checkpoint_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model_state_dict": checkpoint_state})

    class EmptyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_formats = []

        def to(self, *args, **kwargs):
            if "memory_format" in kwargs:
                self.memory_formats.append(kwargs["memory_format"])
            return self

        def load_state_dict(self, state, strict=True):
            assert state is checkpoint_state
            assert strict is True
            return super().load_state_dict(state, strict=strict)

    monkeypatch.setattr(pipeline, "StripModel", EmptyModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Emulate a preceding model that enabled autotuning and disabled TF32.
    # monkeypatch restores the process policy after the test.
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return checkpoint_path


def test_cpu_constructor_preserves_other_models_cuda_policy(monkeypatch, tmp_path):
    checkpoint_path = _prepare_constructor_without_inference(monkeypatch, tmp_path)

    model = pipeline.SynthStrip(weights=checkpoint_path, device="cpu")

    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert model.model.training is False


def test_cuda_constructor_keeps_existing_fixed_algorithm_and_tf32(monkeypatch, tmp_path):
    checkpoint_path = _prepare_constructor_without_inference(monkeypatch, tmp_path)
    model = pipeline.SynthStrip(weights=checkpoint_path, device="cuda:0")
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert model.model.training is False
    assert model.model.memory_formats == []


def test_cuda_configure_precision_false_retains_callers_precision(monkeypatch, tmp_path):
    checkpoint_path = _prepare_constructor_without_inference(monkeypatch, tmp_path)
    pipeline.SynthStrip(weights=checkpoint_path, device="cuda:0", configure_precision=False)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_cpu_one_dnn_constructor_uses_channels_last(monkeypatch, tmp_path):
    checkpoint_path = _prepare_constructor_without_inference(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.backends.mkldnn, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", True)
    model = pipeline.SynthStrip(weights=checkpoint_path, device="cpu")
    assert model._cpu_channels_last is True
    assert model.model.memory_formats == [torch.channels_last_3d]


def test_cpu_constructor_respects_disabled_one_dnn(monkeypatch, tmp_path):
    checkpoint_path = _prepare_constructor_without_inference(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.backends.mkldnn, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    model = pipeline.SynthStrip(weights=checkpoint_path, device="cpu")
    assert model._cpu_channels_last is False
    assert model.model.memory_formats == []
