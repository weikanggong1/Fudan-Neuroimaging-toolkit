import math
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.fnirt.io import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    load_fsl_coefficients,
    make_fsl_coefficient_image,
    save_fsl_coefficients,
)
from fnit.fnirt.spline import fsl_control_shape


ORACLE = (
    Path(__file__).with_name("data")
    / "fsl_coefficient_2203_0_oracle.nii.gz"
)


def _coefficients():
    shape = fsl_control_shape((9, 10, 8), (3, 2, 2))
    flat = np.asarray(
        [
            math.sin(0.17 * (index + 1))
            + 0.03 * ((index % 7) - 3)
            for index in range(math.prod(shape))
        ],
        dtype=np.float32,
    )
    base = flat.reshape(shape[2], shape[1], shape[0]).transpose(2, 1, 0)
    return np.stack((base, 2 * base, -0.5 * base), axis=-1)


def _affine():
    return np.asarray(
        [
            [1.1, 0.0, 0.0, 4.0],
            [0.0, 0.9, 0.0, -3.0],
            [0.0, 0.0, 1.2, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_coefficient_image_matches_fnirt_file_writer_2203_0():
    expected = nib.load(str(ORACLE))
    actual = make_fsl_coefficient_image(
        _coefficients(),
        (9, 10, 8),
        (2.0, 2.5, 3.0),
        (3, 2, 2),
        _affine(),
    )
    np.testing.assert_array_equal(
        np.asarray(actual.dataobj), np.asarray(expected.dataobj)
    )
    assert int(actual.header["intent_code"]) == FSL_CUBIC_SPLINE_COEFFICIENTS
    assert int(actual.header["xyzt_units"]) == int(expected.header["xyzt_units"]) == 10
    np.testing.assert_allclose(
        actual.header["pixdim"][1:4], expected.header["pixdim"][1:4]
    )
    np.testing.assert_allclose(
        [
            actual.header["intent_p1"],
            actual.header["intent_p2"],
            actual.header["intent_p3"],
        ],
        [
            expected.header["intent_p1"],
            expected.header["intent_p2"],
            expected.header["intent_p3"],
        ],
    )
    np.testing.assert_allclose(actual.get_qform(), expected.get_qform())
    np.testing.assert_allclose(actual.get_sform(), expected.get_sform())


def test_coefficient_file_round_trip_preserves_fsl_metadata(tmp_path):
    filename = tmp_path / "coefficients.nii.gz"
    save_fsl_coefficients(
        filename,
        _coefficients(),
        (9, 10, 8),
        (2.0, 2.5, 3.0),
        (3, 2, 2),
        _affine(),
    )
    loaded = load_fsl_coefficients(filename)
    np.testing.assert_array_equal(loaded.coefficients, _coefficients())
    assert loaded.field_shape == (9, 10, 8)
    assert loaded.field_voxel_sizes == (2.0, 2.5, 3.0)
    assert loaded.knot_spacing == (3, 2, 2)
    np.testing.assert_allclose(loaded.affine_forward, _affine())
