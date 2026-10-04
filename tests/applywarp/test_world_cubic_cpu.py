"""Independent SciPy contract for the CPU fMRIPrep query protocol."""
import numpy as np
import pytest
import torch

from fnit._world_resampling import _sample_fmriprep_image_cpu


@pytest.mark.parametrize("threads,chunk_size", [(1, 17), (2, 17), (2, 10000)])
@pytest.mark.parametrize("interpolation,order", [("nearest", 0), ("linear", 1), ("spline", 3)])
def test_cpu_complete_frames_and_chunked_queries_match_scipy(threads, chunk_size, interpolation, order):
    from scipy.ndimage import map_coordinates

    rng = np.random.default_rng(391)
    source = rng.normal(size=(7, 8, 9, 3)).astype(np.float32)
    # Include source edges, zero-extension support, and far outside queries.
    coordinates = rng.uniform(-16, 25, size=(3, 3, 4, 5, 6))
    coordinates[:, :, 0, 0, :3] = np.array([[-.5, 0, 6.5], [-.5, 0, 7.5], [-.5, 0, 8.5]])
    expected = np.stack([
        map_coordinates(source[..., frame], coordinates[frame], order=order,
                        mode="grid-constant", cval=0, prefilter=True,
                        output=np.float32)
        for frame in range(3)
    ])
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        actual = _sample_fmriprep_image_cpu(
            source, torch.from_numpy(coordinates), chunk_size, interpolation,
        ).numpy()
    finally:
        torch.set_num_threads(previous)
    assert actual.shape == (3, 4, 5, 6)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("precision,expected_time_unit", [("fmriprep", "sec"), ("float64", "unknown")])
def test_only_fmriprep_3d_preserves_target_time_flag(precision, expected_time_unit):
    import nibabel as nib
    from fnit.applywarp import TorchApplyWarp, WorldTransformChain

    source = nib.Nifti1Image(np.arange(120, dtype=np.float32).reshape(4, 5, 6), np.eye(4))
    reference = nib.Nifti1Image(np.zeros((4, 5, 6), np.float32), np.eye(4))
    reference.header.set_xyzt_units("mm", "sec")
    image = TorchApplyWarp(device="cpu").apply_world(
        source, WorldTransformChain(reference, np.eye(4), coordinate_precision=precision),
        interpolation="nearest",
    )
    assert image.header.get_xyzt_units() == ("mm", expected_time_unit)
    np.testing.assert_array_equal(image.dataobj, source.dataobj)


def test_fmriprep_cpu_nearest_retains_scipy_half_ties():
    import nibabel as nib
    from fnit.applywarp import TorchApplyWarp, WorldTransformChain

    values = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None], (5, 3, 3)).copy()
    source = nib.Nifti1Image(values, np.eye(4))
    reference_affine = np.eye(4)
    reference_affine[0, 3] = .5
    reference = nib.Nifti1Image(np.zeros((4, 3, 3), np.float32), reference_affine)
    chain = WorldTransformChain(reference, np.eye(4), coordinate_precision="fmriprep")
    image = TorchApplyWarp(device="cpu").apply_world(source, chain, interpolation="nearest")
    np.testing.assert_array_equal(np.asarray(image.dataobj)[:, 1, 1], [1, 2, 3, 4])
