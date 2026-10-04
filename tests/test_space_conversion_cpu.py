"""Targeted CPU equivalence tests; synthetic arrays are not benchmarks."""

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from argparse import Namespace
import importlib.util
from pathlib import Path
from scipy.io import savemat

from fnit._space_conversion_cpu import project_surface_layer
from fnit.space_conversion import _cpu_thread_budget, _save_volume


@pytest.mark.parametrize("frames", [1, 3])
def test_fused_nearest_lookup_matches_torch_at_boundaries(frames):
    shape = (7, 6, 5)
    rng = np.random.default_rng(5201)
    mask = rng.uniform(size=shape).astype(np.float32)
    maps = [rng.integers(0, 17, size=(shape[1], shape[0], shape[2]))
            .astype(np.float32) for _ in range(2)]
    data = [rng.normal(size=(16, frames)).astype(np.float32) for _ in range(2)]
    voxels = rng.uniform(-1, 8, size=(250, 3)).astype(np.float32)
    # Explicit half-voxel ties, just-inside edges and out-of-volume samples.
    edges = np.array([[0.5, 2.5, 1.5], [1.5, 0.5, 2.5], [-0.5, 1, 1],
                      [6.5, 1, 1], [6.49999, 1, 1]], np.float32)
    voxels = np.concatenate((voxels, edges))
    grid = np.ascontiguousarray(2 * voxels / (np.array(shape, np.float32) - 1) - 1)
    tensor_grid = torch.from_numpy(grid).reshape(1, -1, 1, 1, 3)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 1, 0)))[None, None]
    valid = F.grid_sample(mask_tensor, tensor_grid, mode="nearest", align_corners=True)[0, 0].ravel() > 0.5
    expected = torch.zeros((len(grid), frames), dtype=torch.float32)
    for vertex_map, values in zip(maps, data):
        volume = torch.from_numpy(np.ascontiguousarray(vertex_map.transpose(2, 0, 1)))[None, None]
        indices = F.grid_sample(volume, tensor_grid, mode="nearest", align_corners=True)[0, 0].ravel().long()
        selected = valid & (indices > 0)
        expected[selected] += torch.from_numpy(values)[indices[selected] - 1]
    actual = project_surface_layer(grid, mask, maps[0], maps[1], data[0], data[1])
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("label,dtype", [(False, np.float32), (True, np.int32)])
def test_volume_save_does_not_inherit_integer_reference_dtype(tmp_path, label, dtype):
    reference = nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), np.eye(4))
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    if not label:
        values = values / np.float32(7.7)
    else:
        values = values.astype(np.int32)
    path = tmp_path / "cortex.nii.gz"
    _save_volume(values, reference, path, label)
    loaded = nib.load(path)
    assert loaded.get_data_dtype() == np.dtype(dtype)
    np.testing.assert_array_equal(np.asarray(loaded.dataobj), values)
    np.testing.assert_array_equal(loaded.affine, reference.affine)


@pytest.mark.parametrize("frames", [1, 2])
def test_official_mat_bridge_preserves_axes_and_single_frame(tmp_path, frames):
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_multimodal_cpu_space.py"
    spec = importlib.util.spec_from_file_location("_space_benchmark_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = np.arange(24 * frames, dtype=np.float32).reshape(2, 3, 4, frames)
    matlab = expected.transpose(1, 0, 2, 3)
    if frames == 1:
        matlab = matlab[..., 0]
    mat = tmp_path / "official.mat"
    savemat(mat, {"volume": matlab})
    reference = tmp_path / "reference.nii.gz"
    output = tmp_path / "output.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), np.eye(4)), reference)
    module._bridge(Namespace(kind="reverse-output", mat=str(mat), reference=str(reference),
                             volume=str(output), label=False))
    np.testing.assert_array_equal(np.asarray(nib.load(output).dataobj),
                                  expected[..., 0] if frames == 1 else expected)


def test_cpu_call_preserves_numba_mask_on_failure(monkeypatch):
    from numba import get_num_threads, set_num_threads

    previous = get_num_threads()
    try:
        if previous < 2:
            pytest.skip("This process has a one-thread Numba ceiling")
        set_num_threads(2)
        monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
        with pytest.raises(RuntimeError, match="projection failed"):
            with _cpu_thread_budget():
                assert get_num_threads() == 1
                raise RuntimeError("projection failed")
        assert get_num_threads() == 2
        monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
        with _cpu_thread_budget():
            assert get_num_threads() == 2
        assert get_num_threads() == 2
    finally:
        set_num_threads(previous)


def test_surface_copy_cli_does_not_require_volume_compute_libraries(tmp_path):
    """A fresh CLI can validate/copy GIFTI while Torch/MAT readers are absent."""
    import os
    import subprocess
    import sys

    sources = []
    for hemi in ("L", "R"):
        path = tmp_path / f"{hemi}.func.gii"
        image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            np.arange(2562, dtype=np.float32), intent="NIFTI_INTENT_SHAPE")])
        nib.save(image, path)
        sources.append(path)
    arguments = ["--source-space", "fsaverage", "--target-space", "fsaverage",
                 "--source-density", "3k", "--target-density", "3k",
                 "--left", str(sources[0]), "--right", str(sources[1]),
                 "--assets-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                 "--device", "cpu"]
    code = """
import importlib.abc
import json
import sys
class MissingVolumeLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname == 'scipy.io':
            raise ImportError('Volume compute library unavailable: ' + fullname)
sys.meta_path.insert(0, MissingVolumeLibraries())
from fnit.space_conversion import main
main(json.loads(sys.argv[1]))
assert 'torch' not in sys.modules
assert 'scipy.io' not in sys.modules
"""
    import json
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-c", code, json.dumps(arguments)], env=env,
                   check=True, capture_output=True, text=True)
    for source, hemi in zip(sources, ("L", "R")):
        assert (tmp_path / "out" / f"{hemi}.fsaverage.3k.func.gii").read_bytes() == source.read_bytes()
