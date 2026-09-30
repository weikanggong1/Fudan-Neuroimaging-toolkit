"""FSL output/header regressions; arrays here are unit fixtures, not benchmarks."""
import nibabel as nib
import numpy as np
import pytest

from fnit.flirt import core


def test_registration_fills_missing_qform_from_reference_sform():
    affine = np.diag([-2., 2., 2., 1.])
    moving = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), affine)
    fixed = nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), affine)
    fixed.set_qform(affine, 0)
    fixed.set_sform(affine, 2)
    result = core._output_image(moving.get_fdata(dtype=np.float32), moving, fixed, np.eye(4))
    assert int(result.header["qform_code"]) == 2
    np.testing.assert_allclose(result.get_qform(), fixed.get_sform())


def test_applyxfm_retains_reference_forms_and_pixdim_with_sform_shear():
    # Independently executed FSL 6.0.7.22 identity-applyxfm header oracle:
    # reference qform=0/sform=2, pixdim=2, output qform=0/sform=2, pixdim=2.
    affine = np.array([[-2., .5, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
    moving = nib.Nifti1Image(np.ones((7, 8, 9), dtype=np.float32), affine)
    fixed = nib.Nifti1Image(np.zeros_like(moving.dataobj), affine)
    for image in (moving, fixed):
        image.header.set_zooms((2, 2, 2))
    result = core.TorchFLIRT(device="cpu").applyxfm(moving, fixed, init=np.eye(4))
    assert int(result.moved.header["qform_code"]) == 0
    assert int(result.moved.header["sform_code"]) == 2
    assert result.moved.header.get_zooms() == (2, 2, 2)
    np.testing.assert_allclose(result.moved.get_sform(), affine)


def test_applyxfm_preserves_distinct_reference_qform_and_sform():
    sform = np.diag([-2., 2., 2., 1.])
    qform = sform.copy()
    qform[:3, 3] = (1, 2, 3)
    moving = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), sform)
    fixed = nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), sform)
    fixed.set_qform(qform, 1)
    fixed.set_sform(sform, 2)
    result = core.TorchFLIRT(device="cpu").applyxfm(moving, fixed, init=np.eye(4))
    np.testing.assert_allclose(result.moved.get_qform(), qform)
    np.testing.assert_allclose(result.moved.get_sform(), sform)


def test_applyxfm_outside_field_uses_edge_background():
    data = np.full((7, 8, 9), 13, dtype=np.float32)
    data[2:-2, 2:-2, 2:-2] = 100
    affine = np.diag([-1., 1., 1., 1.])
    moving = nib.Nifti1Image(data, affine)
    fixed = nib.Nifti1Image(np.zeros_like(data), affine)
    matrix = np.eye(4)
    matrix[0, 3] = 2
    result = core.TorchFLIRT(device="cpu").applyxfm(moving, fixed, init=matrix)
    np.testing.assert_array_equal(result.moved.get_fdata()[:2], 13)


def test_integer_mask_output_keeps_fractional_interpolation():
    data = np.zeros((7, 8, 9), dtype=np.int16)
    data[2:-2, 2:-2, 2:-2] = 1
    affine = np.diag([-1., 1., 1., 1.])
    moving = nib.Nifti1Image(data, affine)
    fixed = nib.Nifti1Image(np.zeros_like(data), affine)
    matrix = np.eye(4)
    matrix[0, 3] = .5
    result = core.TorchFLIRT(device="cpu").applyxfm(moving, fixed, init=matrix)
    assert result.moved.get_data_dtype() == np.dtype("float32")
    assert .5 in result.moved.get_fdata()


@pytest.mark.parametrize("affine, pixdim", (
    (np.array([[-2., .5, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]),
     (2., 2., 2.)),
    # A real public image's affine rounding made ceil(min sampling) become 2
    # while its NIfTI header says 1 mm, incorrectly skipping the final level.
    (np.diag([-1.0000000213913103, 1.0000000213913103,
              1.0000000213913103, 1.]), (1., 1., 1.)),
))
def test_registration_uses_header_pixdim_and_retains_integer_dtype(monkeypatch, affine, pixdim):
    data = np.arange(7 * 8 * 9, dtype=np.int16).reshape(7, 8, 9)
    moving = nib.Nifti1Image(data, affine)
    fixed = nib.Nifti1Image(np.zeros_like(data), affine)
    for image in (moving, fixed):
        image.header.set_zooms(pixdim)

    class IdentityEngine:
        def __init__(self, *args, **kwargs):
            assert args[4] == args[5] == pixdim
            self.device = kwargs["device"]
            self.execution = kwargs["execution"]
            self.candidate_batch_size = kwargs["candidate_batch_size"]
            self.memory_budget_gb = kwargs["memory_budget_gb"]
            self.phase_timings = {}
            self.cost_evaluations = 0
            self.batch_evaluations = self.host_result_transfers = 0
            self.phase_cost_evaluations = {}
            self._active_phase = None

        def run(self, *args, **kwargs):
            return 0., np.eye(4)

        phase = core._DefaultFLIRTEngine.phase

    monkeypatch.setattr(core, "_DefaultFLIRTEngine", IdentityEngine)
    result = core.TorchFLIRT(device="cpu")(moving, fixed)
    assert result.moved.get_data_dtype() == np.dtype("int16")
    assert result.moved.header.get_zooms() == pixdim
    assert int(result.moved.header["qform_code"]) == 2
    assert int(result.moved.header["sform_code"]) == 2
    np.testing.assert_array_equal(result.moved.dataobj, data)
