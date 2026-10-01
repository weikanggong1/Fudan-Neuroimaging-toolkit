"""Coordinate-composition checks for EPI-to-T1-to-MNI resampling."""

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.normalization import resample_world


def test_float64_pull_is_not_implicitly_quantized_before_spline_sampling(tmp_path):
    from scipy.ndimage import map_coordinates
    x, y, z = np.indices((9, 5, 5))
    data = ((x - 4) * 1e8 - 12345679 + 7 * y + 11 * z).astype(np.float32)
    source_path, target_path = tmp_path / "source.nii.gz", tmp_path / "target.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source_path)
    affine = np.diag([.01, .01, .01, 1.])
    affine[:3, 3] = [4., 1.5, 1.5]
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.float32), affine), target_path)
    target = nib.load(target_path)
    coordinates = nib.affines.apply_affine(
        target.affine, np.indices(target.shape).reshape(3, -1).T,
    ).T
    outputs = []
    for dtype in (np.float64, np.float32):
        pull = np.zeros((*target.shape, 3), dtype=dtype)
        pull[..., 0] = .123456789123
        field_path = tmp_path / "pull.nii.gz"
        nib.save(nib.Nifti1Image(pull, target.affine), field_path)
        output = resample_world(
            source_path, target_path, np.eye(4), tmp_path / "output.nii.gz",
            pre_affine_pull_ras=field_path, interpolation="spline", device="cpu",
        )
        actual = np.asarray(nib.load(output).dataobj)
        expected = map_coordinates(
            data, coordinates + pull.reshape(-1, 3).T,
            order=3, mode="grid-constant",
        ).reshape(target.shape)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
        outputs.append(actual)
    # The two files encode distinct transforms. The original premature f4
    # conversion incorrectly made their sampled outputs identical.
    assert np.max(np.abs(outputs[0] - outputs[1])) > .1


def test_fmriprep_precision_composes_hmc_in_voxel_space(tmp_path):
    from scipy.ndimage import map_coordinates
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(118)
    data = rng.normal(size=(10, 11, 12, 2)).astype(np.float32)
    source_affine = np.eye(4)
    source_affine[:3, :3] = Rotation.from_euler("xyz", (2., -3., 4.), degrees=True).as_matrix() @ np.diag([1.2, 1.3, 1.4])
    source_affine[:3, 3] = [-72.3167, 16.7263, -52.8391]
    target_affine = source_affine.copy()
    target_affine[:3, 3] += [.43, .22, .36]
    source_path, target_path = tmp_path/"source.nii.gz", tmp_path/"target.nii.gz"
    source = nib.Nifti1Image(data, source_affine)
    source.header.set_zooms((1.2, 1.3, 1.4, .735))
    source.header.set_xyzt_units(t="sec")
    nib.save(source, source_path)
    nib.save(nib.Nifti1Image(np.zeros((5, 6, 7), np.float32), target_affine), target_path)
    source, target = nib.load(source_path), nib.load(target_path)
    matrix = np.eye(4)
    matrix[:3, 3] = [.11, -.16, .27]
    motion = np.repeat(np.eye(4)[None], 2, axis=0)
    motion[1, :3, :3] = Rotation.from_euler("xyz", (.1, .2, -.15), degrees=True).as_matrix()
    voxels = np.indices(target.shape).reshape(3, -1).T
    world = nib.affines.apply_affine(target.affine, voxels).astype(np.float32)
    fixed_world = nib.affines.apply_affine(matrix, world)
    common_voxel = nib.affines.apply_affine(np.linalg.inv(source.affine),
                                            fixed_world.astype(np.float32))
    expected = []
    for frame in range(2):
        voxel_hmc = np.linalg.inv(source.affine) @ motion[frame] @ source.affine
        coords = nib.affines.apply_affine(voxel_hmc, common_voxel).T.reshape(3, *target.shape)
        expected.append(map_coordinates(data[..., frame], coords, order=3, mode="grid-constant"))
    path = resample_world(source_path, target_path, matrix, tmp_path/"out.nii.gz",
                           motion_pull_world=motion, interpolation="spline",
                           coordinate_precision="fmriprep", device="cpu")
    np.testing.assert_allclose(np.asarray(nib.load(path).dataobj), np.stack(expected, axis=-1),
                                rtol=0, atol=1e-6)


def test_fmriprep_dense_field_preserves_deformation_grid_precision():
    import torch
    from scipy.ndimage import map_coordinates
    from fnit.fmri.normalization import _fmriprep_dense_world
    shape = (5, 6, 7)
    affine = np.array([[1.23121, .021, 0, -84.3927], [0, 1.34003, .014, 31.6473],
                       [.002, 0, 1.47021, -53.4321], [0, 0, 0, 1]], dtype=np.float64)
    index = np.indices(shape).reshape(3, -1)
    world = nib.affines.apply_affine(affine, index.T).T
    rng = np.random.default_rng(119)
    delta = rng.normal(scale=.1, size=(*shape, 3)).astype(np.float32)
    field = nib.Nifti1Image(delta, affine)
    # Use the actual saved-header affine precision to define both grid and field.
    world = nib.affines.apply_affine(field.affine, index.T).T
    deformations = world + delta.reshape(-1, 3).T
    for shift in (0., .07):
        query = world + shift
        mapped = _fmriprep_dense_world(torch.tensor(query), field,
                                       torch.tensor(delta.reshape(-1, 3).T, dtype=torch.float64),
                                       torch.tensor(index, dtype=torch.float64), 37).numpy()
        if shift == 0:
            np.testing.assert_array_equal(mapped, deformations)
        else:
            coordinates = nib.affines.apply_affine(np.linalg.inv(field.affine), query.T.astype(np.float32)).T
            expected = np.vstack([map_coordinates(deformations[axis].reshape(shape), coordinates,
                                                   order=3, mode="constant", cval=np.nan)
                                   for axis in range(3)])
            missing = np.isnan(expected)
            expected[missing] = query.astype(np.float32)[missing]
            np.testing.assert_allclose(mapped, expected, rtol=0, atol=1e-10)


def test_world_affine_and_pull_compose_before_one_4d_interpolation(tmp_path):
    x, y, z = np.indices((7, 7, 7))
    source_data = np.stack((x + 10 * y + 100 * z,
                            1000 + x + 10 * y + 100 * z), axis=-1).astype(np.float32)
    source = nib.Nifti1Image(source_data, np.eye(4))
    source.header.set_zooms((1, 1, 1, 0.8))
    source.header.set_xyzt_units(t="sec")
    source_path = tmp_path / "source.nii.gz"
    nib.save(source, str(source_path))
    reference = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(reference, str(reference_path))
    pull_data = np.zeros((4, 4, 4, 3), dtype=np.float32)
    pull_data[..., 1] = 1
    pull_path = tmp_path / "pull.nii.gz"
    nib.save(nib.Nifti1Image(pull_data, np.eye(4)), str(pull_path))
    output_mask = np.ones((4, 4, 4), dtype=np.uint8)
    output_mask[0] = 0
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(output_mask, np.eye(4)), str(mask_path))
    affine = np.eye(4)
    affine[0, 3] = 1
    output = resample_world(
        source_path, reference_path, affine, tmp_path / "output.nii.gz",
        pre_affine_pull_ras=pull_path, output_mask=mask_path,
        batch_size=1, device="cpu",
    )
    actual = nib.load(str(output))
    expected = source_data[1:5, 1:5, 0:4].copy()
    expected[0] = 0
    np.testing.assert_allclose(np.asarray(actual.dataobj), expected, atol=1e-5)
    assert actual.shape == (4, 4, 4, 2)
    assert actual.header.get_zooms()[3] == pytest.approx(0.8)


def test_world_pull_rejects_wrong_reference_grid(tmp_path):
    source = tmp_path / "source.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    pull = tmp_path / "pull.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), str(source))
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), str(reference))
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 4, 3)), np.eye(4)), str(pull))
    with pytest.raises(ValueError, match="reference grid"):
        resample_world(source, reference, np.eye(4), tmp_path / "out.nii.gz",
                       pre_affine_pull_ras=pull, device="cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_spline_matches_independent_periodic_oracle_and_preserves_frames(tmp_path, device):
    import torch
    from scipy.ndimage import map_coordinates
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(31)
    data = rng.normal(size=(11, 12, 10, 3)).astype(np.float32)
    source = tmp_path / "source.nii.gz"
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_zooms((1, 1, 1, .735))
    image.header.set_xyzt_units(t="sec")
    nib.save(image, source)
    reference = tmp_path / "reference.nii.gz"
    shape = (6, 7, 5)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), reference)
    affine = np.eye(4)
    affine[:3, 3] = (1.37, 1.61, .42)
    coordinates = np.indices(shape, dtype=float) + affine[:3, 3, None, None, None]
    expected = np.stack([map_coordinates(data[..., frame], coordinates, order=3,
                                        mode="grid-wrap") for frame in range(3)], axis=-1)
    results = []
    for batch in (1, 3):
        path = resample_world(source, reference, affine, tmp_path / f"out-{batch}.nii.gz",
                              interpolation="spline", boundary="periodic",
                              batch_size=batch, device=device)
        actual = nib.load(path)
        np.testing.assert_allclose(np.asarray(actual.dataobj), expected, atol=3e-5, rtol=2e-5)
        assert actual.header.get_zooms()[3] == pytest.approx(.735)
        assert actual.header.get_xyzt_units()[1] == "sec"
        results.append(np.asarray(actual.dataobj))
    np.testing.assert_allclose(*results, atol=3e-5, rtol=2e-5)


def test_spline_preserves_mask_and_out_of_field_zeros(tmp_path):
    source, reference, mask = (tmp_path / f"{name}.nii.gz" for name in ("source", "reference", "mask"))
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), reference)
    valid = np.ones((5, 5, 5), dtype=np.uint8)
    valid[:, 0] = 0
    nib.save(nib.Nifti1Image(valid, np.eye(4)), mask)
    affine = np.eye(4)
    affine[0, 3] = -.5
    output = resample_world(source, reference, affine, tmp_path / "out.nii.gz",
                            output_mask=mask, interpolation="spline", boundary="periodic",
                            device="cpu")
    values = np.asarray(nib.load(output).dataobj)
    assert not values[0].any()
    assert not values[:, 0].any()
    np.testing.assert_allclose(values[1:, 1:], 1, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("interpolation", ["linear", "nearest", "spline"])
def test_oblique_identity_preserves_boundary_voxels_and_time_axis(tmp_path, device, interpolation):
    import torch
    from scipy.spatial.transform import Rotation
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(37)
    data = rng.normal(size=(13, 15, 11, 2)).astype(np.float32)
    affine = np.eye(4)
    affine[:3, :3] = (
        Rotation.from_euler("xyz", (9, 14, -7), degrees=True).as_matrix()
        @ np.diag((-2.0, 2.1, 2.2))
    )
    affine[:3, 3] = (-92.314, 41.824, -66.392)
    source = tmp_path / "source.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    image = nib.Nifti1Image(data, affine)
    image.header.set_zooms((2.0, 2.1, 2.2, .735))
    image.header.set_xyzt_units(xyz="mm", t="sec")
    nib.save(image, source)
    nib.save(nib.Nifti1Image(np.zeros(data.shape[:3], dtype=np.float32), affine), reference)
    path = resample_world(
        source, reference, np.eye(4), tmp_path / "output.nii.gz",
        interpolation=interpolation, batch_size=1, device=device,
    )
    actual = nib.load(path)
    # Equality on the entire image covers all six FOV faces, including corners.
    np.testing.assert_allclose(np.asarray(actual.dataobj), data, atol=2e-5, rtol=2e-5)
    assert actual.get_data_dtype() == np.dtype(np.float32)
    assert actual.shape == data.shape
    np.testing.assert_allclose(actual.affine, nib.load(reference).affine, atol=1e-7)
    assert actual.header.get_zooms()[3] == pytest.approx(.735)
    assert actual.header.get_xyzt_units()[1] == "sec"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("interpolation", ["linear", "nearest", "spline"])
@pytest.mark.parametrize("shift", [-2e-6, -5e-7, 5e-7, 2e-6])
def test_boundary_roundoff_is_clamped_but_genuine_outside_is_zero(
    tmp_path, device, interpolation, shift,
):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    shape = (5, 6, 7)
    source = tmp_path / "source.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), reference)
    matrix = np.eye(4)
    matrix[0, 3] = shift
    path = resample_world(
        source, reference, matrix, tmp_path / "output.nii.gz",
        interpolation=interpolation, boundary="periodic", device=device,
    )
    expected = np.ones(shape, dtype=np.float32)
    if abs(shift) > 1e-6:
        expected[0 if shift < 0 else -1] = 0
    np.testing.assert_allclose(np.asarray(nib.load(path).dataobj), expected, atol=2e-6)
