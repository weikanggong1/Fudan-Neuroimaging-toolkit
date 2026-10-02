"""Native BOLD resolution and oblique anatomical FOV regressions."""
import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.sampling_reference import native_bold_sampling_reference
from fnit.fmri.surface_prepare import prepare_t1w_surface_geometry


def test_oblique_reference_matches_locked_nilearn_grid(tmp_path):
    # Independently measured with Nilearn 0.11.1 / NiWorkflows 1.14.4.
    affine = np.array([[0, -1.1, 0, -20], [1.2, 0, .1, 17],
                       [0, 0, 1.3, -13], [0, 0, 0, 1.]])
    fixed = tmp_path / "t1.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    moving = tmp_path / "bold.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((17, 19, 15), dtype="f4"), affine), fixed)
    nib.save(nib.Nifti1Image(np.pad(np.ones((9, 11, 7), dtype="u1"), 4), affine), mask)
    nib.save(nib.Nifti1Image(np.ones((9, 8, 7), dtype="f4"),
                           np.diag([-2.4, 2.4, 3., 1.])), moving)
    output = native_bold_sampling_reference(fixed, moving, mask, tmp_path / "ref.nii.gz")
    image = nib.load(output)
    assert image.shape == (9, 9, 7)
    assert np.allclose(image.header.get_zooms(), (2.4, 2.4, 3.))
    assert np.allclose(image.affine[:3, 3], (-39.8, 17., -13.), atol=2e-6)
    assert int(image.header["sform_code"]) == int(image.header["qform_code"]) == 2


def _recon(tmp_path):
    root = tmp_path / "recon"
    (root / "mri").mkdir(parents=True)
    (root / "surf").mkdir()
    nib.save(nib.MGHImage(np.zeros((3, 3, 3), dtype="f4"), np.eye(4)), root / "mri/orig.mgz")
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    for hemi in ("lh", "rh"):
        for name, values in (("white", vertices), ("pial", vertices + .4),
                             ("graymid", vertices + .15), ("midthickness", vertices + .12)):
            nib.freesurfer.io.write_geometry(root / "surf" / f"{hemi}.{name}", values, faces)
    return root, vertices


def test_existing_midthickness_and_fsnative_affine_are_used(tmp_path):
    root, vertices = _recon(tmp_path)
    forward = np.eye(4)
    forward[:3, 3] = (7, -3, 2)
    geometry = prepare_t1w_surface_geometry(root, tmp_path / "output", fsnative_to_t1w=forward)
    orig = nib.load(root / "mri/orig.mgz")
    transform = forward @ orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    expected = nib.affines.apply_affine(transform, vertices + .12)
    np.testing.assert_allclose(nib.load(geometry.left.midthickness).darrays[0].data,
                               expected, atol=1e-6)
    assert geometry.left.midthickness_source.name == "lh.midthickness"


def test_missing_midthickness_does_not_silently_use_vertex_mean(tmp_path):
    root, _ = _recon(tmp_path)
    for name in ("midthickness", "graymid"):
        (root / "surf" / f"lh.{name}").unlink()
    with pytest.raises(FileNotFoundError, match="midthickness or lh.graymid"):
        prepare_t1w_surface_geometry(root, tmp_path / "output")
