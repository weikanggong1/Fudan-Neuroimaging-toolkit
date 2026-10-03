"""Failure gates for the real-data routing witness; no GPU/model execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture(scope="module")
def tool():
    path = Path(__file__).resolve().parents[1] / "tools/validate_fmri_resampler_routing.py"
    spec = importlib.util.spec_from_file_location("fmri_routing_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(api, frames=9):
    records = []
    for role in ("mask_mni", "clean_mni", "preproc_t1w", "preproc_mni"):
        shape = [3, 4, 5] if role == "mask_mni" else [3, 4, 5, frames]
        motion = ({"shape": [frames, 4, 4], "nonidentity_frames": frames - 1}
                  if role.startswith("preproc") else None)
        records.append({"api": api, "public_chain": "resample_world" not in api,
                        "input": {"shape": shape},
                        "returned": {"shape": shape, "dtype": "float32", "affine": {"sha256": "a"}},
                        "chain": {"reference": {"shape": shape[:3], "affine": {"sha256": "a"}},
                                  "pull": None if role == "preproc_t1w" else {"nonzero": True},
                                  "motion": motion,
                                  "coordinate_precision": "fmriprep" if motion else "float64"},
                        "interpolation": "nearest" if role == "mask_mni" else "spline",
                        "boundary": "periodic" if role == "clean_mni" else "grid-constant",
                        "output_mask_present": role == "clean_mni",
                        "caller_chain": [{"module": "fnit.fmri.end_to_end"}]})
    return records


@pytest.mark.parametrize("backend", ["fnirt", "synthmorph"])
def test_equal_images_cannot_substitute_for_actual_public_routes(tool, backend):
    legacy = _records("fnit.fmri.normalization.resample_world")
    assert tool.routing_gate(legacy, backend=backend, variant="baseline", expected_frames=9)["passed"]
    assert not tool.routing_gate(legacy, backend=backend, variant="candidate", expected_frames=9)["passed"]


def test_fnirt_requires_both_actual_run_world_and_apply_world(tool):
    run = _records("fnit.applywarp.TorchApplyWarp.run_world")
    apply = _records("fnit.applywarp.TorchApplyWarp.apply_world")
    assert not tool.routing_gate(run, backend="fnirt", variant="candidate", expected_frames=9)["passed"]
    assert tool.routing_gate(run + apply, backend="fnirt", variant="candidate", expected_frames=9)["passed"]


@pytest.mark.parametrize("corruption", ["frames", "identity_motion", "zero_pull", "no_caller", "not_public_chain"])
def test_synthmorph_requires_full_frames_and_real_composed_transforms(tool, corruption):
    records = _records("fnit.synthmorph.apply_transform")
    assert tool.routing_gate(records, backend="synthmorph", variant="candidate", expected_frames=9)["passed"]
    if corruption == "frames":
        records[3]["returned"]["shape"][-1] = 8
    elif corruption == "identity_motion":
        records[3]["chain"]["motion"]["nonidentity_frames"] = 0
    elif corruption == "zero_pull":
        records[3]["chain"]["pull"]["nonzero"] = False
    elif corruption == "no_caller":
        records[3]["caller_chain"] = []
    else:
        records[3]["public_chain"] = False
    assert not tool.routing_gate(records, backend="synthmorph", variant="candidate", expected_frames=9)["passed"]


def test_witness_forwards_original_objects_without_modifying_arrays(tool, tmp_path):
    from fnit._world_resampling import WorldTransformChain

    values = np.zeros((3, 4, 5, 9), dtype=np.float32)
    values[0, 0, 0, 0] = -0.0
    source = nib.Nifti1Image(values, np.eye(4))
    target = nib.Nifti1Image(np.zeros((3, 4, 5), np.float32), np.eye(4))
    motion = np.broadcast_to(np.eye(4), (9, 4, 4)).copy()
    motion[1:, 0, 3] = .25
    chain = WorldTransformChain(target, np.eye(4), motion_pull_world=motion,
                                coordinate_precision="fmriprep")
    before = values.tobytes(), motion.tobytes()
    seen = []

    def delegate(image, transformation, *, method="spline", boundary="grid-constant", frame_chunk_size=8):
        seen.append((image, transformation, method, boundary, frame_chunk_size))
        return image

    spy = tool.RouteSpy(tmp_path, SimpleNamespace())
    spy._public_chain_type = WorldTransformChain
    spy._callers = lambda: []
    wrapped = spy._observe(delegate, "fnit.synthmorph.apply_transform")
    assert wrapped(source, chain, frame_chunk_size=4) is source
    assert seen[0][0] is source and seen[0][1] is chain
    assert seen[0][-1] == 4 and before == (values.tobytes(), motion.tobytes())
    assert spy.records[0]["input"]["shape"][-1] == 9
    assert spy.records[0]["chain"]["motion"]["nonidentity_frames"] == 8


def test_whole_run_peak_survives_component_resets_without_changing_them(tool):
    class Cuda:
        def __init__(self):
            self.allocated, self.reserved = 14, 18
            self.resets = []

        def max_memory_allocated(self):
            return self.allocated

        def max_memory_reserved(self):
            return self.reserved

        def reset_peak_memory_stats(self, *args, **kwargs):
            self.resets.append((args, kwargs))
            self.allocated, self.reserved = 2, 3

    cuda = Cuda()
    with tool.WholeRunMemory(cuda) as memory:
        cuda.reset_peak_memory_stats("cuda:0")
        cuda.allocated, cuda.reserved = 5, 8
    assert (memory.allocated, memory.reserved) == (14, 18)
    assert cuda.resets == [(("cuda:0",), {})] and memory.reset_calls == 1


def test_synthmorph_comparison_still_rejects_signed_zero_changes(tool, tmp_path):
    helpers = tool._load_helpers()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    values = np.zeros((2, 2, 2, 3), dtype=np.float32)
    for folder in (first, second):
        nib.save(nib.Nifti1Image(values, np.eye(4)), folder / "volume.nii.gz")
    assert tool.compare_trees(first, second, backend="synthmorph", helpers=helpers)["all_equal"]
    altered = values.copy()
    altered[0, 0, 0, 2] = -0.0
    nib.save(nib.Nifti1Image(altered, np.eye(4)), second / "volume.nii.gz")
    result = tool.compare_trees(first, second, backend="synthmorph", helpers=helpers)
    assert not result["all_equal"]
    assert next(iter(result["outputs"].values()))["changed_values_including_signed_zero"] == 1
