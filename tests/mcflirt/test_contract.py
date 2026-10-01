"""MCFLIRT 的参数、dtype 和文件合同；真实图像精度另在 validation 测量。"""
import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.mcflirt import TorchMCFLIRT
from fnit.mcflirt.core import _normcorr_reduce, mcflirt_output_dtype
from fnit.fmri.motion import matrices_to_mcflirt_parameters


def _series(dtype=np.uint16):
    axes = np.meshgrid(*(np.arange(10) for _ in range(3)), indexing="ij")
    first = (100 + axes[0] * 3 + axes[1] * 2 + axes[2]).astype(dtype)
    data = np.stack((first, first + 5, first + 10), axis=-1)
    image = nib.Nifti1Image(data, np.diag((-2.4, 2.4, 2.4, 1)))
    image.header.set_zooms((2.4, 2.4, 2.4, .735))
    image.header.set_xyzt_units(t="sec")
    reference = nib.Nifti1Image(first, image.affine)
    return image, reference


def test_unsigned_short_output_and_parameter_files(tmp_path):
    image, reference = _series()
    prefix = tmp_path / "corrected"
    result = TorchMCFLIRT(device="cpu").run(
        image, reference, output=prefix, mats=True, plots=True,
        stage_iterations=(0, 0, 0), interpolation="linear")
    saved = nib.load(result.output_paths["corrected"])
    assert saved.get_data_dtype() == np.dtype(np.int32)
    np.testing.assert_array_equal(np.asarray(saved.dataobj), np.asarray(image.dataobj))
    np.testing.assert_allclose(saved.header.get_zooms(), image.header.get_zooms())
    assert saved.header.get_xyzt_units()[1] == "sec"
    np.testing.assert_array_equal(result.matrices, np.repeat(np.eye(4)[None], 3, 0))
    np.testing.assert_allclose(np.loadtxt(result.output_paths["parameters"]), 0, atol=1e-12)
    assert len(list((tmp_path / "corrected.mat").glob("MAT_*"))) == 3
    with pytest.raises(FileExistsError):
        TorchMCFLIRT(device="cpu").run(image, reference, output=prefix)


def test_default_reference_is_the_original_middle_frame():
    image, _ = _series(np.float32)
    result = TorchMCFLIRT(device="cpu").run(image, stage_iterations=(0, 0, 0))
    np.testing.assert_array_equal(np.asarray(result.reference.dataobj), np.asarray(image.dataobj)[..., 1])
    np.testing.assert_array_equal(result.matrices[1], np.eye(4))
    np.testing.assert_allclose(result.parameters, matrices_to_mcflirt_parameters(result.matrices, result.reference))


def test_ncc_preserves_native_running_count_instead_of_pearson():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 10
    y = x * 1.2 + 2
    w = np.ones_like(x)
    # Independent literal source accumulation. Native num/numA are not
    # reset at row/plane boundaries, unlike the five intensity sums.
    count = count_a = count_b = np.float32(0)
    total = np.zeros(5, dtype=np.float32)
    for plane in range(2):
        plane_sums = np.zeros(5, dtype=np.float32)
        for row in range(3):
            row_sums = np.zeros(5, dtype=np.float32)
            for column in range(4):
                xx, yy = x[plane, row, column], y[plane, row, column]
                count = np.float32(count + 1)
                row_sums += np.array([xx, xx * xx, yy, yy * yy, xx * yy], dtype=np.float32)
            count_a = np.float32(count_a + count)
            plane_sums += row_sums
        count_b = np.float32(count_b + count_a)
        total += plane_sums
    sx, sx2, sy, sy2, sxy = total
    covariance = np.float32(sxy / np.float32(count_b - 1) - sx * sy / (count_b * count_b))
    vx = np.float32(sx2 / np.float32(count_b - 1) - sx * sx / (count_b * count_b))
    vy = np.float32(sy2 / np.float32(count_b - 1) - sy * sy / (count_b * count_b))
    expected = np.float32(1 - abs(covariance / np.sqrt(vx) / np.sqrt(vy)))
    actual = _normcorr_reduce(torch.tensor(x), torch.tensor(y), torch.tensor(w))
    np.testing.assert_allclose(float(actual), float(expected), atol=1e-7)
    assert float(actual) > 1e-5  # Ordinary Pearson on y=1.2*x+2 would be 1.


def test_unsupported_short_fov_is_explicit():
    image = nib.Nifti1Image(np.ones((10, 10, 2, 3), np.float32), np.eye(4))
    with pytest.raises(NotImplementedError, match="2D"):
        TorchMCFLIRT(device="cpu").run(image)


@pytest.mark.parametrize("input_dtype,output_dtype", [
    (np.uint8, np.uint8), (np.int8, np.uint8), (np.int16, np.int16),
    (np.uint16, np.int32), (np.int32, np.int32), (np.uint32, np.float32),
    (np.int64, np.float32), (np.uint64, np.float32),
    (np.float32, np.float32), (np.float64, np.float64),
])
def test_native_closest_templated_type(input_dtype, output_dtype):
    assert mcflirt_output_dtype(input_dtype) == np.dtype(output_dtype)
