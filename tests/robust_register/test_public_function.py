"""Thin public function keeps the mature CPU implementation and lifecycle."""
from importlib import import_module
import inspect

import pytest

from fnit.robust_register import CPURegistration, FNIT_mri_robust_register


def test_single_stage_forwards_all_defaults_and_closes(monkeypatch):
    calls = []
    contexts = []
    sentinel = object()
    original_init = CPURegistration.__init__

    def initialize(self):
        original_init(self)
        contexts.append(self)

    def compute(self, source, target, **parameters):
        self._ensure_open()
        calls.append((source, target, parameters))
        return sentinel

    monkeypatch.setattr(CPURegistration, "__init__", initialize)
    monkeypatch.setattr(CPURegistration, "cpu_robust_register", compute)
    assert FNIT_mri_robust_register("source", "target") is sentinel
    parameters = inspect.signature(FNIT_mri_robust_register).parameters
    expected = {key: value.default for key, value in parameters.items()
                if key not in {"source", "target", "stage_directory"}}
    assert calls == [("source", "target", expected)]
    assert len(contexts) == 1 and contexts[0]._closed


def test_pair_forwards_exact_stage_and_overrides(monkeypatch, tmp_path):
    calls = []
    sentinel = object()

    def pair(self, source, target, **parameters):
        self._ensure_open()
        calls.append((self, source, target, parameters))
        return sentinel

    monkeypatch.setattr(CPURegistration, "cpu_robust_rigid_affine", pair)
    directory = tmp_path / "new-pair"
    result = FNIT_mri_robust_register(
        "source", "target", mode="rigid-affine", stage_directory=directory,
        saturation=37.0, spatial_chunk_size=31)
    assert result is sentinel
    context, source, target, parameters = calls[0]
    assert context._closed
    assert (source, target) == ("source", "target")
    assert parameters["stage_directory"] == directory
    assert parameters["saturation"] == 37.0
    assert parameters["spatial_chunk_size"] == 31
    assert "mode" not in parameters
    assert not directory.exists()  # the wrapper does not create scientific output


@pytest.mark.parametrize("options", [
    {"mode": "unknown"},
    {"mode": "rigid-affine"},
    {"mode": "affine", "stage_directory": "unused"},
])
def test_invalid_dispatch_rejected_before_model_construction(monkeypatch, options):
    def unexpected(self):
        raise AssertionError("constructed context for invalid dispatch")
    monkeypatch.setattr(CPURegistration, "__init__", unexpected)
    with pytest.raises(ValueError):
        FNIT_mri_robust_register("missing-source", "missing-target", **options)


def test_native_cpu_only_validation_is_preserved():
    import torch
    initialized = torch.cuda.is_initialized()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(ValueError, match="only device=cpu"):
        FNIT_mri_robust_register("missing-source", "missing-target", device="cuda:0")
    assert torch.cuda.is_initialized() == initialized
    assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_exception_propagates_and_context_is_closed(monkeypatch):
    contexts = []

    def fail(self, *args, **kwargs):
        contexts.append(self)
        raise LookupError("mature registration failed")

    monkeypatch.setattr(CPURegistration, "cpu_robust_register", fail)
    with pytest.raises(LookupError, match="mature registration failed"):
        FNIT_mri_robust_register("source", "target", mode="affine")
    assert contexts[0]._closed


def test_existing_stage_directory_failure_is_preserved(tmp_path):
    with pytest.raises(FileExistsError):
        FNIT_mri_robust_register(
            "missing-source", "missing-target", mode="rigid-affine", stage_directory=tmp_path)
