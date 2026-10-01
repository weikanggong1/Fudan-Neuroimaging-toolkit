"""FSL physical voxel sizes come from pixdim, independently of the sform."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np

from fnit.flirt.coordinates import flirt_to_world_affine
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT, standalone


def _image(shear, zooms, shape=(12, 12, 12)):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[0, 1] = shear
    image = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    image.header.set_zooms(zooms)
    return image


def test_standalone_fsl_affine_uses_stored_pixdim(monkeypatch, tmp_path):
    moving = _image(.7, (1.0, 1.5, 2.0))
    fixed = _image(.2, (2.0, 2.0, 2.5))
    fsl_matrix = np.array([[.99, .04, .01, 2], [0, 1.01, .02, -3], [.02, 0, 1.02, 1], [0, 0, 0, 1]])
    captured = []

    class CaptureFNIRT:
        def __init__(self, **kwargs):
            pass

        def __call__(self, moving, fixed, matrix, **kwargs):
            captured.append(matrix.matrix)
            return SimpleNamespace(coefficient_image=fixed, moved=fixed, nonlinear_jacobian=fixed)

    monkeypatch.setattr(standalone, "TorchFNIRT", CaptureFNIRT)
    monkeypatch.setattr(standalone, "_write_outputs_atomic", lambda *args: None)
    standalone.run_fnirt(moving, fixed, fsl_matrix, config="default", cout=tmp_path / "warp.nii.gz", device="cpu")
    expected = flirt_to_world_affine(fsl_matrix, moving.affine, fixed.affine, moving.shape, fixed.shape, moving.header.get_zooms()[:3], fixed.header.get_zooms()[:3])
    np.testing.assert_allclose(captured[0], expected, rtol=0, atol=1e-14)


def test_registration_preserves_header_physical_sizes_in_coefficient_and_outputs():
    moving = _image(.7, (1.0, 1.5, 2.0))
    fixed = _image(.2, (2.0, 2.0, 2.5))
    fsl_matrix = np.eye(4)
    forward_world = flirt_to_world_affine(fsl_matrix, moving.affine, fixed.affine, moving.shape, fixed.shape, moving.header.get_zooms()[:3], fixed.header.get_zooms()[:3])
    config = GMFNIRTConfig(subsampling=(1,), maximum_iterations=(0,), input_fwhm_mm=(0,), reference_fwhm_mm=(0,), regularization=(0,), estimate_intensity=(False,), apply_reference_mask=(False,), implicit_input_mask=False, implicit_reference_mask=False)
    result = TorchFNIRT(device="cpu", config=config)(moving, fixed, forward_world)
    np.testing.assert_allclose(result.coefficient_image.get_sform(), fsl_matrix, atol=1e-7, rtol=0)
    np.testing.assert_array_equal([result.coefficient_image.header[name] for name in ("intent_p1", "intent_p2", "intent_p3")], fixed.header.get_zooms())
    for image in (result.moved, result.nonlinear_jacobian, result.full_pull_jacobian, result.modulated_gm):
        np.testing.assert_array_equal(image.header.get_zooms()[:3], fixed.header.get_zooms())
