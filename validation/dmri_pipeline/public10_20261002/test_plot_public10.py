"""Plot coordinates and range contracts; fixtures are not benchmark evidence."""

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "plot_public10", Path(__file__).with_name("plot_public10.py")
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_canonical_orientation_only_flips_existing_voxels():
    values = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    image = nib.Nifti1Image(values, affine)
    canonical, matrix = plot.canonical_values(image, values)
    np.testing.assert_array_equal(canonical, values[::-1])
    assert matrix[0, 0] == 2 and matrix[0, 3] == -4


def test_nearest_world_plane_records_actual_coordinate():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[2, 3] = 12
    index, actual_z, _ = plot.slice_geometry((3, 4, 5), affine, 16.6)
    assert index == 2 and actual_z == 16
    with pytest.raises(ValueError, match="outside"):
        plot.slice_geometry((3, 4, 5), affine, 22)


def test_oblique_grid_does_not_get_resampled_for_display():
    affine = np.eye(4)
    affine[0, 1] = 0.1
    values = np.ones((3, 4, 5), np.float32)
    with pytest.raises(ValueError, match="axis-aligned"):
        plot.canonical_values(nib.Nifti1Image(values, affine), values)


@pytest.mark.parametrize("branch", ["tbss", "mmorf"])
def test_plot_uses_anonymous_hashes_fixed_ranges_and_singleton_original(tmp_path, branch):
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [2, -2, 12]
    shape = (3, 4, 5)
    mask = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), affine), mask)
    candidate, original = tmp_path / "candidate", tmp_path / "original"
    (candidate / "registration" / "standard").mkdir(parents=True)
    for name, first, second in (("FA", 1.25, 0.5), ("MD", 0.002, 0.001), ("ICVF", 0.8, 0.8)):
        destination = candidate / "registration" / "standard" / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.full(shape, first, np.float32), affine), destination)
        reference = plot.original_map_path(original, branch, name)
        reference.parent.mkdir(parents=True, exist_ok=True)
        original_shape = shape + (1,) if branch == "tbss" else shape
        nib.save(nib.Nifti1Image(np.full(original_shape, second, np.float32), affine), reference)
    image_path = tmp_path / "brain.png"
    report = plot.make_figure(candidate_dir=candidate, original_dir=original,
                              branch=branch, mask_template=mask, output=image_path,
                              z_mm=16.6, case_label="case01")
    assert image_path.stat().st_size > 0
    assert report["display"]["actual_world_z_mm"] == 16
    assert report["display"]["imshow_interpolation"] == "nearest"
    assert report["maps"]["FA"]["panels"]["FNIT"]["above_range_fraction"] == 1
    assert report["maps"]["FA"]["panels"]["Absolute difference"]["above_range_fraction"] == 1
    assert report["maps"]["ICVF"]["panels"]["Absolute difference"]["above_range_fraction"] == 0
    assert report["maps"]["FA"]["sources"]["Original"]["singleton_volume_squeezed"] == (branch == "tbss")
    assert str(tmp_path) not in json.dumps(report)
    assert json.loads(image_path.with_suffix(".json").read_text())["figure"]["sha256"] == report["figure"]["sha256"]
