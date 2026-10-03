import numpy as np
import pytest

from fnit.recon_all.volmask_python import _inside_mesh, _projected_edge, ribbon_arrays


FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                  [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])


def cube(low, high, x_shift=0):
    a, b = low, high
    xyz = np.array([[a, a, a], [b, a, a], [b, b, a], [a, b, a],
                    [a, a, b], [b, a, b], [b, b, b], [a, b, b]], dtype=np.float32)
    xyz[:, 0] += x_shift
    return xyz, FACES


def test_nested_surfaces_make_bilateral_white_and_gray_masks():
    surfaces = {hemi: {"white": cube(1.5, 3.5, shift),
                       "pial": cube(0.5, 4.5, shift)}
                for hemi, shift in (("lh", 0), ("rh", 6))}
    ribbon, left, right = ribbon_arrays((12, 6, 6), np.eye(4), surfaces)
    assert ribbon[2, 2, 3] == 2
    assert ribbon[1, 2, 3] == 3
    assert ribbon[8, 2, 3] == 41
    assert ribbon[7, 2, 3] == 42
    assert ribbon[0, 0, 0] == 0
    assert left[1, 2, 3] == 1 and right[7, 2, 3] == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_diagonal_preserves_every_interior_voxel(reverse):
    # y+delta == z+delta hits a face diagonal exactly: closed ownership counted
    # two triangles per crossing and silently removed 16 of 64 interior voxels.
    vertices, faces = cube(0.5, 4.5)
    faces = faces[:, ::-1].copy() if reverse else faces.copy()
    inside, odd, overflow = _inside_mesh(vertices, faces, (6, 6, 6))
    expected = np.zeros((6, 6, 6), np.uint8)
    expected[1:5, 1:5, 1:5] = 1
    np.testing.assert_array_equal(inside, expected)
    assert odd == overflow == 0


def test_shifted_ray_bbox_matches_analytic_slanted_closed_prism():
    vertices, faces = cube(0.5, 4.5)
    vertices[[0, 4], 1] = np.float32(2.000005)
    inside, odd, overflow = _inside_mesh(vertices, faces.astype(np.int32), (6, 6, 6))
    x, y, z = np.indices((6, 6, 6), dtype=np.float64)
    lower_y = float(vertices[0, 1]) + (x - 0.5) / 4 * (0.5 - float(vertices[0, 1]))
    expected = ((x > 0.5) & (x < 4.5) & (y + 1e-5 >= lower_y)
                & (y + 1e-5 < 4.5) & (z + 1e-5 > 0.5) & (z + 1e-5 < 4.5))
    np.testing.assert_array_equal(inside, expected.astype(np.uint8))
    assert odd == overflow == 0


def test_canonical_shared_edge_is_bitwise_opposite_near_boundary():
    # Saved float32 endpoints adjacent to the real CON01 duplicate crossing.
    first = np.array([144.53469848632812, 77.88948059082031, 175.08154296875])
    second = np.array([144.1132049560547, 78.65279388427734, 174.5184783935547])
    forward = _projected_edge(first, second, 125972, 111418, 78.00001, 175.00001)
    reverse = _projected_edge(second, first, 111418, 125972, 78.00001, 175.00001)
    assert np.float64(forward).tobytes() == np.float64(-reverse).tobytes()


def test_thin_closed_prism_keeps_nonzero_projected_faces():
    vertices, faces = cube(0.5, 4.5)
    # Projected areas are about 4e-10; discarding every det below 1e-8
    # silently erased the four inside samples with no odd-ray warning.
    vertices[:, 1:] = np.where(vertices[:, 1:] == 0.5, 2.0, 2.00002)
    inside, odd, overflow = _inside_mesh(vertices, faces, (6, 6, 6))
    expected = np.zeros((6, 6, 6), np.uint8)
    expected[1:5, 2, 2] = 1
    np.testing.assert_array_equal(inside, expected)
    assert odd == overflow == 0


def test_tangent_vertex_does_not_change_even_odd_inside():
    offset = 2 + 1e-5
    vertices = np.array([[2, offset, offset], [1, 3, offset],
                         [3, offset, 3], [2, 3, 3]], np.float64)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    inside, odd, overflow = _inside_mesh(vertices, faces, (5, 5, 5))
    assert not inside[:, 2, 2].any()
    assert odd == overflow == 0


def test_open_mesh_still_rejects_unpaired_ray():
    vertices, faces = cube(0.5, 4.5)
    surfaces = {hemi: {name: (vertices, faces[:-1] if hemi == "lh" and name == "pial" else faces)
                       for name in ("white", "pial")} for hemi in ("lh", "rh")}
    with pytest.raises(ValueError, match=r"lh\.pial has [1-9]\d* odd"):
        ribbon_arrays((6, 6, 6), np.eye(4), surfaces)


def test_crossing_capacity_still_rejects_overflow():
    vertices, faces = cube(0.5, 4.5)
    many_vertices = np.concatenate([vertices + [i * 5, 0, 0] for i in range(64)]).astype(np.float32)
    many_faces = np.concatenate([faces + i * 8 for i in range(64)]).astype(np.int32)
    surfaces = {hemi: {name: (many_vertices, many_faces) for name in ("white", "pial")}
                for hemi in ("lh", "rh")}
    with pytest.raises(ValueError, match=r"0 odd and [1-9]\d* overflowing rays"):
        ribbon_arrays((326, 6, 6), np.eye(4), surfaces)
