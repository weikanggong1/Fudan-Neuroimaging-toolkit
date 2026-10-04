import inspect

import nibabel as nib
import numba
import numpy as np
import pytest
import torch

import fnit
from fnit import gems
from fnit.gems import pipeline
from fnit.gems.context import SubregionContext


def test_public_entry_is_unique_and_uses_the_canonical_pipeline():
    assert fnit.segment_4_subregions is gems.segment_4_subregions is pipeline.segment_4_subregions
    assert "segment_4_subregions" in gems.__all__
    for name in ("segment_subregions", "segment_nuclei", "prepare_nuclei_atlas"):
        assert name not in gems.__all__
        assert not hasattr(gems, name) and not hasattr(fnit, name)
    parameters = inspect.signature(fnit.segment_4_subregions).parameters
    assert parameters["structures"].default == "all"
    assert parameters["optimization"].default == "fast"
    assert parameters["threads"].default == 4
    assert {"auto_initialize", "em_iterations", "deform_iterations"}.isdisjoint(parameters)
    assert not hasattr(pipeline, "_segment_atlas_packs")


@pytest.mark.parametrize("structures", ("synthetic", "hippo-left", "hippo-right", []))
def test_only_supported_structure_families_enter_the_pipeline(structures):
    with pytest.raises(ValueError, match="structures"):
        pipeline.segment_4_subregions("unused.nii.gz", structures=structures, device="cpu")


@pytest.mark.parametrize("threads", (0, -1, True, False, 1.5, "4", None))
def test_invalid_threads_are_rejected_before_resources_are_read(threads):
    with pytest.raises(ValueError, match="positive integer"):
        pipeline.segment_4_subregions("unused.nii.gz", threads=threads, device="cpu")


@pytest.mark.parametrize("threads", (1, 4, 8))
def test_thread_configuration_reaches_shared_context(tmp_path, monkeypatch, threads):
    image = nib.Nifti1Image(np.ones((8, 8, 8), np.float32), np.eye(4))
    directory = tmp_path / "brainstem"
    directory.mkdir()
    (directory / "AtlasMesh.gz").touch()
    configured = []
    before_torch = torch.get_num_threads()
    before_numba = numba.get_num_threads()
    original_set_threads = torch.set_num_threads
    def set_threads(value):
        configured.append(value)
        original_set_threads(value)
    monkeypatch.setattr(torch, "set_num_threads", set_threads)

    class PreparedEnough(Exception):
        pass

    def prepare(*args, **kwargs):
        assert configured[-1] == threads
        assert torch.get_num_threads() == threads
        assert numba.get_num_threads() == threads
        assert kwargs["need_coarse"] and not kwargs["need_parc"]
        assert kwargs["device"] == torch.device("cpu")
        raise PreparedEnough

    monkeypatch.setattr(SubregionContext, "prepare", prepare)
    with pytest.raises(PreparedEnough):
        pipeline.segment_4_subregions(image, tmp_path, structures="brainstem", threads=threads,
                                     device="cpu")
    assert torch.get_num_threads() == before_torch
    assert numba.get_num_threads() == before_numba


def test_cpu_scope_restores_threads_on_success():
    before_torch = torch.get_num_threads()
    before_numba = numba.get_num_threads()
    requested = 1 if before_torch != 1 or before_numba != 1 else 2
    @pipeline._cpu_thread_scoped
    def finished(*, device="cpu", threads=4):
        assert torch.get_num_threads() == threads
        assert numba.get_num_threads() == threads
        return "finished"
    assert finished(device="cpu", threads=requested) == "finished"
    assert torch.get_num_threads() == before_torch
    assert numba.get_num_threads() == before_numba


def test_cpu_scope_preserves_torch_budget_above_numba_capacity(monkeypatch):
    before_torch = torch.get_num_threads()
    before_numba = numba.get_num_threads()
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 1)
    @pipeline._cpu_thread_scoped
    def finished(*, device="cpu", threads=4):
        assert torch.get_num_threads() == threads
        assert numba.get_num_threads() == 1
        raise RuntimeError("after preparation")
    with pytest.raises(RuntimeError, match="after preparation"):
        finished(device="cpu", threads=8)
    assert torch.get_num_threads() == before_torch
    assert numba.get_num_threads() == before_numba


def test_cuda_scope_does_not_enter_cpu_budget(monkeypatch):
    import fnit.recon_all.thread_budget as budget
    def forbidden(**kwargs):
        raise AssertionError("CUDA path must preserve its original thread policy")
    monkeypatch.setattr(budget, "thread_budget", forbidden)
    @pipeline._cpu_thread_scoped
    def finished(*, device="cuda:0", threads=4):
        return device, threads
    assert finished(device="cuda:0", threads=8) == ("cuda:0", 8)
