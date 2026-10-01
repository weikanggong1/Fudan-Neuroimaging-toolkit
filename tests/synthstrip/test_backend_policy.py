"""SynthStrip fixes convolution selection while retaining default TF32."""

import torch

from fnit.synthstrip import pipeline


def test_constructor_disables_autotuning_after_other_models_enable_it(monkeypatch, tmp_path):
    # No weights, downloads, GPU, or inference are needed to verify the policy.
    checkpoint_path = tmp_path / "unloaded_synthstrip.1.pt"
    checkpoint_state = {}
    monkeypatch.setattr(pipeline, "resolve_weights", lambda name, explicit: checkpoint_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model_state_dict": checkpoint_state})

    class EmptyModel(torch.nn.Module):
        def load_state_dict(self, state, strict=True):
            assert state is checkpoint_state
            assert strict is True
            return super().load_state_dict(state, strict=strict)

    monkeypatch.setattr(pipeline, "StripModel", EmptyModel)
    # Emulate a preceding model that enabled autotuning and disabled TF32.
    # monkeypatch restores the process policy after the test.
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    model = pipeline.SynthStrip(weights=checkpoint_path, device="cpu")

    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert model.model.training is False
