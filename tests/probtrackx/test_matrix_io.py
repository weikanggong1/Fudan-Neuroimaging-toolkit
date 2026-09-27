"""Exact text layout and index order from the vendored FSL ptx2 source."""

import numpy as np
import pytest

from fnit.probtrackx.matrix_io import ordered_voxels, write_dot, write_volume_coords


def test_ordered_voxels_scans_x_fastest_then_y_then_z():
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    np.testing.assert_array_equal(ordered_voxels(mask), [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ])
    with pytest.raises(ValueError, match="3D"):
        ordered_voxels(np.ones((2, 2)))


def test_dot_is_one_based_column_major_and_always_ends_with_shape(tmp_path):
    path = tmp_path / "fdt_matrix1.dot"
    write_dot(path, {(1, 2): 3.5, (0, 0): 2, (0, 2): 0,
                     (1, 0): 1.25}, (2, 3))
    assert path.read_text() == "1  1  2\n2  1  1.25\n2  3  3.5\n2  3  0\n"
    write_dot(path, {}, (2, 3))
    assert path.read_text() == "2  3  0\n"
    with pytest.raises(ValueError, match="outside"):
        write_dot(path, {(2, 0): 1}, (2, 3))


def test_volume_coordinate_table_preserves_roi_order_and_nifti_x_flip(tmp_path):
    first = np.zeros((3, 2, 2), dtype=np.uint8)
    second = first.copy()
    first[0, 0, 0] = first[1, 1, 0] = 1
    second[1, 1, 0] = second[2, 0, 1] = 1
    path = tmp_path / "coords_for_fdt_matrix1"
    write_volume_coords(path, [first, second], flip_x=True)
    assert path.read_text() == (
        "2  0  0  0  1  \n"
        "1  1  0  0  2  \n"
        "1  1  0  1  2  \n"
        "0  0  1  1  3  \n"
    )
    write_volume_coords(path, second, with_roi=False)
    assert path.read_text() == "1  1  0  \n2  0  1  \n"
