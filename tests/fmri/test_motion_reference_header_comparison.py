"""NCC输入不能只核对解码像素和affine，stored pixdim也属于合同。"""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[2] / 'validation/fmri/compare_matched_pipeline.py'
SPEC = importlib.util.spec_from_file_location('fnit_matched_compare', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_identical_pixels_affine_can_have_different_scaled_mm_headers(tmp_path):
    values = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
    first = nib.Nifti1Image(values.astype(np.float32), np.eye(4))
    second = nib.Nifti1Image(values, np.eye(4))
    first.header['pixdim'][3] = np.nextafter(np.float32(1), np.float32(2))
    candidate, reference = tmp_path / 'candidate.nii', tmp_path / 'reference.nii'
    nib.save(first, candidate)
    nib.save(second, reference)
    result = MODULE.motion_reference_headers(
        {'candidate': str(candidate), 'reference': str(reference)}, tmp_path)
    assert result['decoded_float32_values_bitwise_equal']
    assert result['affine_exact_equal']
    assert not result['stored_pixdim_float32_bits_equal']
    assert result['headers']['candidate']['storage_dtype'] == 'float32'
    assert result['headers']['reference']['storage_dtype'] == 'int16'
    assert result['stored_pixdim_candidate_minus_reference_mm'][2] > 0
