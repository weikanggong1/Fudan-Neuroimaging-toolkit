"""CPU storage/query partitioning must preserve existing interpolation bits."""
import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import fnit.applywarp.core as core
from fnit.applywarp import TorchApplyWarp


def _legacy(data, grid):
    return F.grid_sample(data[None], grid, mode="bilinear", padding_mode="border",
                         align_corners=True)[0]


@pytest.mark.parametrize("shape", [(11, 13, 9), (1, 5, 7)])
@pytest.mark.parametrize("handedness", [-1, 1])
def test_diagonal_grid_and_direct_grid_storage_preserve_float_bits(shape, handedness):
    matrix = np.diag([handedness * 1.123457, 2.345678, 3.123459, 1.])
    matrix[:3, 3] = [13.456789, -.01723456, 0]
    axes = torch.meshgrid(*(torch.arange(size, dtype=torch.float64) for size in shape), indexing="ij")
    voxels = torch.stack(axes).reshape(3, -1)
    transform = torch.as_tensor(matrix)
    expected = transform[:3, :3] @ voxels + transform[:3, 3:4]
    actual = core._spatial_grid(shape, matrix, "cpu")
    assert torch.equal(actual.view(torch.int64), expected.view(torch.int64))
    coordinates = actual.reshape(3, *shape)
    expected_grid = core._to_grid(coordinates, shape).to(torch.float32)
    actual_grid = core._to_float32_grid_cpu(coordinates, shape)
    assert torch.equal(actual_grid.view(torch.int32), expected_grid.view(torch.int32))


def test_direct_grid_preserves_extreme_and_nonfinite_coordinate_semantics():
    values = torch.tensor([-0.0, 0.0, -1e300, 1e300, float("nan"),
                           float("inf"), -float("inf"), .499999999999], dtype=torch.float64)
    coordinates = values.reshape(1, 2, 2, 2).expand(3, -1, -1, -1).clone()
    for shape in [(1, 13, 17), (11, 13, 17)]:
        expected = core._to_grid(coordinates, shape).to(torch.float32)
        actual = core._to_float32_grid_cpu(coordinates, shape)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("channels,dtype", [(1, torch.float32), (7, torch.float32), (3, torch.float64)])
def test_cpu_spatial_batches_retain_bits_and_caller_thread_budget(threads, channels, dtype):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        generator = torch.Generator().manual_seed(714)
        # An odd output count exercises trailing dummy-query removal.
        source = torch.randn((channels, 13, 14, 11), dtype=dtype, generator=generator)
        grid = torch.rand((1, 41, 43, 39, 3), dtype=dtype, generator=generator) * 2.4 - 1.2
        baseline = _legacy(source, grid)
        actual = core._sample_linear_cpu(source, grid)
        bits = torch.int32 if dtype == torch.float32 else torch.int64
        assert torch.equal(actual.view(bits), baseline.view(bits))
        assert torch.get_num_threads() == threads
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("frame_chunk_size", [None, 2])
def test_complete_cpu_4d_plan_keeps_metadata_mask_and_signed_zero(monkeypatch, frame_chunk_size):
    generator = np.random.default_rng(107)
    values = generator.normal(size=(12, 11, 10, 5)).astype(np.float32)
    source = nib.Nifti1Image(values, np.diag([-1.5, 2, 2.5, 1]))
    source.header.set_zooms((1.5, 2, 2.5, 0.725))
    reference = nib.Nifti1Image(np.zeros((39, 43, 41), np.float32), source.affine)
    reference.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"cpu-storage-contract"))
    premat = np.eye(4)
    premat[:3, 3] = [.123, -.27, .319]
    warper = TorchApplyWarp("cpu", frame_chunk_size=frame_chunk_size)
    actual = warper(source, reference, premat=premat, output_dtype="float")
    monkeypatch.setattr(core, "_sample_linear_cpu", _legacy)
    baseline = warper(source, reference, premat=premat, output_dtype="float")
    np.testing.assert_array_equal(np.asarray(actual.image.dataobj).view(np.uint32),
                                  np.asarray(baseline.image.dataobj).view(np.uint32))
    np.testing.assert_array_equal(actual.valid_mask, baseline.valid_mask)
    assert actual.image.header.binaryblock == baseline.image.header.binaryblock
    assert actual.image.header.extensions[0].get_content() == b"cpu-storage-contract"
    assert actual.qc == baseline.qc

