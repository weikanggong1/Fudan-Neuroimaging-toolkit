"""SynthMorph-to-FSL field coordinates and image resampling."""

import nibabel as nib
import numpy as np
import pytest
from types import SimpleNamespace

from fnit import cli
from fnit._transforms import DenseWarp
from fnit.applywarp import TorchApplyWarp
from fnit.flirt.coordinates import voxel_to_fsl_scaled_mm
from fnit.synthmorph import apply_transform, convert_warp_to_fsl


@pytest.mark.parametrize("moving_x_sign", (-1, 1))
def test_fsl_warp_preserves_pull_map_and_resampling(tmp_path, moving_x_sign):
    shape = (10, 9, 8)
    moving_affine = np.diag((moving_x_sign * 1.2, 1.1, 1.3, 1))
    moving_affine[:3, 3] = (4, -3, 2)
    fixed_affine = moving_affine.copy()
    fixed_affine[0, 0] *= -1
    fixed_affine[0, 3] += moving_affine[0, 0] * (shape[0] - 1)
    coordinates = np.indices(shape, dtype=np.float32)
    moving = nib.Nifti1Image(
        coordinates[0] + 3 * coordinates[1] + 5 * coordinates[2],
        moving_affine,
    )
    fixed = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), fixed_affine)
    displacement = np.zeros((*shape, 3), dtype=np.float32)
    displacement[..., 0] = 0.24 + 0.02 * coordinates[1]
    displacement[..., 1] = -0.22
    displacement[..., 2] = 0.13
    native = DenseWarp(displacement, source=moving, target=fixed)

    converted = convert_warp_to_fsl(native, moving=moving, fixed=fixed)
    native_path = tmp_path / "native.mgz"
    native.save(native_path)
    converted_from_file = convert_warp_to_fsl(
        native_path, moving=moving, fixed=fixed
    )
    np.testing.assert_allclose(
        np.asarray(converted_from_file.dataobj), np.asarray(converted.dataobj),
        atol=2e-5,
    )
    assert int(converted.header["intent_code"]) == 2006
    np.testing.assert_allclose(converted.affine, fixed.affine)
    target_voxel = np.array([3, 4, 5, 1], dtype=np.float64)
    source_world = fixed_affine @ target_voxel
    source_world[:3] += displacement[3, 4, 5]
    moving_fsl = voxel_to_fsl_scaled_mm(
        moving.affine, moving.shape, moving.header.get_zooms()[:3]
    )
    fixed_fsl = voxel_to_fsl_scaled_mm(
        fixed.affine, fixed.shape, fixed.header.get_zooms()[:3]
    )
    expected = moving_fsl @ np.linalg.inv(moving.affine) @ source_world
    expected -= fixed_fsl @ target_voxel
    np.testing.assert_allclose(converted.dataobj[3, 4, 5], expected[:3], atol=2e-6)

    saved = tmp_path / "fsl_warp.nii.gz"
    converted.save(saved)
    fsl_image = nib.load(saved)
    assert int(fsl_image.header["intent_code"]) == 2006
    original = apply_transform(moving, native)
    applied = TorchApplyWarp("cpu")(
        moving, fixed, warp=fsl_image, interpolation="trilinear",
        output_dtype="float",
    )
    np.testing.assert_allclose(
        np.asarray(applied.image.dataobj)[1:-1, 1:-1, 1:-1],
        np.asarray(original.dataobj)[1:-1, 1:-1, 1:-1],
        atol=3e-5,
    )
    assert applied.qc["warp_convention"] == "relative"


def test_fsl_warp_requires_matching_moving_geometry():
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    other = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32),
                            np.diag((2, 1, 1, 1)))
    warp = DenseWarp(np.zeros((4, 5, 6, 3), dtype=np.float32),
                     source=image, target=image)
    with pytest.raises(ValueError, match="source/target geometry"):
        convert_warp_to_fsl(warp, moving=other, fixed=image)


def test_cli_writes_fsl_warp_and_rejects_affine(tmp_path, monkeypatch):
    import fnit.synthmorph as synthmorph

    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    moving = tmp_path / "moving.nii.gz"
    fixed = tmp_path / "fixed.nii.gz"
    output = tmp_path / "fsl_warp.nii.gz"
    nib.save(image, moving)
    nib.save(image, fixed)
    native = DenseWarp(np.zeros((4, 5, 6, 3), dtype=np.float32),
                       source=image, target=image)

    class FakeModel:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            return SimpleNamespace(moved=None, fixed_moved=None,
                                   transform=native, inverse=None)

    monkeypatch.setattr(synthmorph, "SynthMorph", FakeModel)
    cli.main(["synthmorph", str(moving), str(fixed), "--fsl-warp", str(output)])
    assert int(nib.load(output).header["intent_code"]) == 2006
    with pytest.raises(SystemExit):
        cli.main(["synthmorph", str(moving), str(fixed), "--model", "affine",
                  "--fsl-warp", str(output)])
