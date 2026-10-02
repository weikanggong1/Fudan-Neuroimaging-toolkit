"""World sampling public contracts and volume backend routing."""
import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.applywarp import TorchApplyWarp, WorldTransformChain
from fnit._world_resampling import _validate_world_memory_budget
from fnit.fmri import end_to_end


@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("boundary", ["grid-constant", "periodic"])
def test_world_spline_motion_matches_independent_scipy(tmp_path, frames, boundary):
    from scipy.ndimage import map_coordinates, spline_filter

    rng = np.random.default_rng(617)
    values = rng.normal(size=(9, 10, 11, frames)).astype(np.float32)
    source_affine = np.array([[-1.2, .13, 0, 12], [0, 1.3, .07, -5],
                              [.04, 0, 1.4, -8], [0, 0, 0, 1.]])
    source = nib.Nifti1Image(values, source_affine)
    source.header.set_xyzt_units("mm", "sec")
    source.header.set_zooms((*source.header.get_zooms()[:3], .735))
    target_affine = source_affine.copy()
    target_affine[:3, 3] = nib.affines.apply_affine(source_affine, [2, 2, 2])
    target = nib.Nifti1Image(np.zeros((3, 4, 5), np.float32), target_affine)
    target.set_qform(target_affine, 0)
    target.set_sform(target_affine, 2)
    target.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"world-contract"))
    affine = np.eye(4)
    affine[:3, 3] = [.11, -.14, .09]
    displacement = np.zeros((*target.shape, 3), dtype=np.float64)
    displacement[..., 0] = .123456789123
    field = nib.Nifti1Image(displacement, target_affine)
    motion = np.repeat(np.eye(4)[None], frames, axis=0)
    motion[:, 0, 1] = np.linspace(0, .011, frames)
    motion[:, 2, 3] = np.linspace(0, .031, frames)
    mask = nib.Nifti1Image(np.ones(target.shape, np.uint8), target_affine)
    mask.dataobj[0, 0, 0] = 0
    chain = WorldTransformChain(target, affine, field, motion)
    result = TorchApplyWarp(device="cpu").apply_world(
        source, chain, interpolation="spline", boundary=boundary,
        output_mask=mask, batch_size=2, spatial_chunk_size=17,
    )
    world = nib.affines.apply_affine(target.affine, np.indices(target.shape).reshape(3, -1).T)
    fixed = nib.affines.apply_affine(affine, world + displacement.reshape(-1, 3))
    expected = []
    for frame in range(frames):
        query = nib.affines.apply_affine(np.linalg.inv(source.affine),
                                        nib.affines.apply_affine(motion[frame], fixed)).T
        if boundary == "grid-constant":
            sampled = map_coordinates(values[..., frame], query, order=3,
                                      mode="grid-constant")
        else:
            # The existing periodic path queries FP32 FFT coefficients with
            # query.float(). Keep SciPy's independent recursive prefilter in
            # FP64, but quantize its query to that established FP32 contract.
            coeff = spline_filter(
                values[..., frame], order=3, mode="grid-wrap", output=np.float64,
            )
            periodic_query = query.astype(np.float32)
            sampled = map_coordinates(
                coeff, periodic_query, order=3, mode="grid-wrap", prefilter=False,
            )
        sampled = sampled.reshape(target.shape).astype(np.float32)
        sampled *= np.asarray(mask.dataobj) > .5
        expected.append(sampled)
    # FP32 FFT coefficient construction and cubic weights may differ from
    # SciPy's FP64 recurrence by several float32 ulps. This is an independent
    # algorithm check; the retained FNIT sampler oracle uses exact bit gates.
    tolerance = 5e-6 if boundary == "periodic" else 2e-6
    np.testing.assert_allclose(
        np.asarray(result.dataobj), np.stack(expected, -1), rtol=0, atol=tolerance,
    )
    assert result.shape == (*target.shape, frames)
    assert int(result.header["qform_code"]) == 0
    assert int(result.header["sform_code"]) == 2
    assert result.header.get_zooms()[3] == source.header.get_zooms()[3]
    assert result.header.get_xyzt_units() == ("mm", "sec")
    assert result.header.extensions[0].get_content() == b"world-contract"
    saved = TorchApplyWarp(device="cpu").run_world(
        source, chain, tmp_path / "complete.nii.gz", interpolation="spline",
        boundary=boundary, output_mask=mask, batch_size=2, spatial_chunk_size=17,
    )
    reloaded = nib.load(saved)
    np.testing.assert_array_equal(np.asarray(reloaded.dataobj).view(np.uint32),
                                  np.asarray(result.dataobj).view(np.uint32))
    assert reloaded.header.binaryblock == result.header.binaryblock


def test_world_nearest_retains_even_half_ties():
    data = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None], (5, 3, 3)).copy()
    source = nib.Nifti1Image(data, np.eye(4))
    affine = np.eye(4)
    affine[0, 3] = .5
    reference = nib.Nifti1Image(np.zeros((4, 3, 3), np.float32), affine)
    chain = WorldTransformChain(reference, np.eye(4))
    result = TorchApplyWarp(device="cpu").apply_world(source, chain, interpolation="nearest")
    np.testing.assert_array_equal(np.asarray(result.dataobj)[:, 1, 1], [0, 2, 2, 4])


def test_world_budget_rejects_before_cuda_allocations(monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 19_500_000_000)
    with pytest.raises(RuntimeError, match="20 GB CUDA budget"):
        _validate_world_memory_budget((88, 88, 64, 490), (91, 109, 91),
            batch_size=8, spatial_chunk_size=262144, interpolation="spline",
            boundary="grid-constant", device=torch.device("cuda:0"))


def test_world_rejects_implicit_fsl_transform():
    with pytest.raises(TypeError, match="WorldTransformChain"):
        TorchApplyWarp(device="cpu").apply_world("image.nii.gz", np.eye(4))


@pytest.mark.parametrize("backend", ["fnirt", "synthmorph"])
def test_volume_dispatches_to_declared_public_interface(tmp_path, monkeypatch, backend):
    events = []
    image = nib.Nifti1Image(np.zeros((2, 3, 4, 2), np.float32), np.eye(4))
    reference = nib.Nifti1Image(np.zeros((2, 3, 4), np.float32), np.eye(4))
    motion = np.repeat(np.eye(4)[None], 2, 0)
    output = tmp_path / "out.nii.gz"

    class Apply:
        def __init__(self, *, device):
            assert device == "cpu"
        def run_world(self, source, chain, path, **kwargs):
            events.append(("TorchApplyWarp", chain, kwargs))
            nib.save(image, path)
            return path

    def transform(source, chain, **kwargs):
        events.append(("apply_transform", chain, kwargs))
        return image

    monkeypatch.setattr(end_to_end, "TorchApplyWarp", Apply)
    monkeypatch.setattr(end_to_end, "apply_transform", transform)
    result = end_to_end._resample_final_volume(
        image, reference, np.eye(4), output, backend=backend,
        motion_pull_world=motion, interpolation="spline", boundary="grid-constant",
        coordinate_precision="fmriprep", batch_size=2, spatial_chunk_size=7, device="cpu",
    )
    assert result == output
    assert len(events) == 1
    assert events[0][0] == ("TorchApplyWarp" if backend == "fnirt" else "apply_transform")
    chain, kwargs = events[0][1:]
    assert isinstance(chain, WorldTransformChain)
    assert chain.reference is reference
    assert chain.motion_pull_world is motion
    assert chain.coordinate_precision == "fmriprep"
    assert kwargs["boundary"] == "grid-constant"
    assert kwargs["spatial_chunk_size"] == 7
    assert kwargs.get("interpolation", kwargs.get("method")) == "spline"
