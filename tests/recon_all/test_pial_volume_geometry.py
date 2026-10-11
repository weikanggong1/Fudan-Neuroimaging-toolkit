"""Pial 头几何校验契约；小网格验证接口，真实输入检查另存报告。"""
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np
import pytest

from fnit.recon_all import place_pial_python as stage
from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel


def _image(shape=(4, 4, 4), affine=None):
    return nib.MGHImage(np.ones(shape, dtype=np.uint8),
                        np.eye(4) if affine is None else affine)


def _metadata():
    return dict(head=np.array([20]), valid="1", filename="unit-test.mgz",
        volume=np.array([4, 4, 4]), voxelsize=np.ones(3),
        xras=np.array([1., 0, 0]), yras=np.array([0., 1, 0]),
        zras=np.array([0., 0, 1]), cras=np.zeros(3))


@pytest.mark.parametrize("valid", ["1", "1  # volume info valid"])
def test_legal_grid_reuses_exact_original_transform_without_reading_arrays(valid):
    images = [_image() for _ in range(3)]
    metadata = _metadata()
    metadata["valid"] = valid
    expected = surface_ras_to_voxel(images[0].header, metadata)
    actual = stage._validate_pial_volume_geometry(*images, metadata)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()
    # 表面头允许来自不同有效空间；沿用原变换，不重置或重采样。
    metadata["cras"] = np.array([2., 3., 4.])
    expected = surface_ras_to_voxel(images[0].header, metadata)
    assert stage._validate_pial_volume_geometry(*images, metadata).tobytes() == expected.tobytes()


@pytest.mark.parametrize("index", [1, 2])
def test_shifted_mri_grid_is_rejected(index):
    images = [_image() for _ in range(3)]
    affine = np.eye(4)
    affine[0, 3] = 1
    images[index] = _image(affine=affine)
    with pytest.raises(ValueError, match="MRI grids differ"):
        stage._validate_pial_volume_geometry(*images, _metadata())


@pytest.mark.parametrize("index", [1, 2])
def test_different_shape_is_rejected(index):
    images = [_image() for _ in range(3)]
    images[index] = _image(shape=(3, 4, 4))
    with pytest.raises(ValueError, match="MRI grids differ"):
        stage._validate_pial_volume_geometry(*images, _metadata())


def test_equivalent_mgh_direction_spacing_factorization_uses_same_scanner_grid():
    images = [_image() for _ in range(3)]
    epsilon = np.float32(2 ** -32)
    scale = np.float32(1 - 2 ** -24)
    directions = np.array([[-1, epsilon, 0], [0, 0, -1], [epsilon, scale, 0]], np.float32)
    for image in images:
        image.header["Mdc"] = directions
        image.header["Pxyz_c"] = [0, 0, 0]
        image._affine = image.header.get_affine()
    images[2].header["delta"] = [1, 1, scale]
    images[2].header["Mdc"][2] = [np.float32(epsilon / scale), 1, 0]
    images[2]._affine = images[2].header.get_affine()
    np.testing.assert_array_equal(images[2].affine, images[0].affine)
    assert not np.array_equal(images[2].header.get_vox2ras_tkr(), images[0].header.get_vox2ras_tkr())
    expected = surface_ras_to_voxel(images[0].header, _metadata())
    assert stage._validate_pial_volume_geometry(*images, _metadata()).tobytes() == expected.tobytes()


@pytest.mark.parametrize("index", [0, 1, 2])
def test_non_scalar_volume_is_rejected(index):
    images = [_image() for _ in range(3)]
    images[index] = _image(shape=(4, 4, 4, 2))
    with pytest.raises(ValueError, match="3D volume"):
        stage._validate_pial_volume_geometry(*images, _metadata())


def test_nonfinite_scanner_geometry_is_rejected():
    images = [_image() for _ in range(3)]
    images[0]._affine[0, 3] = np.nan
    with pytest.raises(ValueError, match="geometry must be finite"):
        stage._validate_pial_volume_geometry(*images, _metadata())


@pytest.mark.parametrize("spacing", [[0, 1, 1], [np.nan, 1, 1]])
def test_invalid_voxel_spacing_is_rejected(spacing):
    images = [_image() for _ in range(3)]
    images[0].header["delta"] = spacing
    with pytest.raises(ValueError, match="positive spacing"):
        stage._validate_pial_volume_geometry(*images, _metadata())


@pytest.mark.parametrize("change", ["missing", "invalid", "trailing_text", "nonfinite", "singular"])
def test_invalid_white_volume_geometry_is_rejected(change):
    images = [_image() for _ in range(3)]
    metadata = _metadata()
    if change == "missing":
        metadata.pop("cras")
    elif change == "invalid":
        metadata["valid"] = "0"
    elif change == "trailing_text":
        metadata["valid"] = "1 unsupported"
    elif change == "nonfinite":
        metadata["cras"][0] = np.nan
    else:
        metadata["zras"] = metadata["yras"].copy()
    with pytest.raises(ValueError, match="geometry"):
        stage._validate_pial_volume_geometry(*images, metadata)


@pytest.mark.parametrize("bad_input", ["wm_affine", "aseg_affine", "aseg_shape", "white_nonfinite"])
def test_api_rejects_invalid_input_before_preparation_and_output(tmp_path, monkeypatch, bad_input):
    for folder in ("surf", "mri", "label"):
        (tmp_path / folder).mkdir()
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    if bad_input == "white_nonfinite":
        xyz[0, 0] = np.nan
    fs.write_geometry(str(tmp_path / "surf/lh.white"), xyz,
        np.array([[0, 1, 2]], np.int32), volume_info=_metadata())
    for name in ("cortex", "cortex+hipamyg"):
        (tmp_path / f"label/lh.{name}.label").write_text(
            "#!ascii label\n3\n0 0 0 0 0\n1 1 0 0 0\n2 0 1 0 0\n")
    (tmp_path / "surf/autodet.gw.stats.lh.dat").write_text("MID_GRAY 50\n")
    for name in ("brain.finalsurfs", "wm", "aseg.presurf"):
        shape = (3, 4, 4) if bad_input == "aseg_shape" and name == "aseg.presurf" else (4, 4, 4)
        affine = np.eye(4)
        if (bad_input == "wm_affine" and name == "wm") or \
                (bad_input == "aseg_affine" and name == "aseg.presurf"):
            affine[1, 3] = 1
        nib.save(_image(shape=shape, affine=affine), str(tmp_path / f"mri/{name}.mgz"))
    def forbidden(*args, **kwargs):
        raise AssertionError("expensive preparation/optimization called")
    monkeypatch.setattr(stage, "prepare_placement_volume", forbidden)
    monkeypatch.setattr(stage, "compute_border_values_first_pass", forbidden)
    monkeypatch.setattr(stage, "repair_intersections", forbidden)
    output = tmp_path / "diagnostic/pial"
    with pytest.raises(ValueError, match="MRI grids differ|coordinates must be finite"):
        stage.place_pial_t1(subject=tmp_path, hemisphere="lh", output=output)
    assert not output.exists()
    assert not output.parent.exists()
