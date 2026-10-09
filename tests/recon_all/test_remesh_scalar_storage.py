"""显式Python双精度标量候选保留原平滑指纹和动态remesh。

模拟小网格只校验数学/接口，真实四侧逐heap决策另作benchmark。
"""
import hashlib

import numpy as np
import pytest

from fnit.recon_all.mris_remesh_python import Mesh, remesh_geometry, remesh_surface, smooth
from test_cpu_geometry_order import _mesh


def test_python_float_storage_keeps_frozen_gauss_seidel_fingerprint():
    vertices, faces = _mesh()
    mesh = Mesh([tuple(row) for row in vertices], faces.tolist(), scalar_storage="python")
    assert all(type(value) is float for point in mesh.points for value in point)
    smooth(mesh=mesh, repeats=2)
    assert all(type(value) is float for point in mesh.points for value in point)
    result = np.asarray(mesh.points, dtype="<f8")
    assert hashlib.sha256(result.tobytes()).hexdigest() == (
        "f2cb82c66d99796c772945a6090a47f503f67fe36eeaa2a5a2ec88af0f77fea6")
    np.testing.assert_array_equal(mesh.faces, faces)


@pytest.mark.parametrize("iterations", [0, 1, 3])
def test_explicit_scalar_storage_keeps_ordered_dynamic_geometry_and_input(iterations):
    vertices, faces = _mesh()
    origin = np.asarray([.9123456, -4.31415, 7.0123])
    vertices = (vertices - origin) * np.asarray([1., 1.7, .42]) + origin
    copy_vertices, copy_faces = vertices.copy(), faces.copy()
    default, default_faces = remesh_geometry(vertices=vertices, faces=faces, iterations=iterations)
    numpy, numpy_faces = remesh_geometry(vertices=vertices, faces=faces, iterations=iterations,
                                         scalar_storage="numpy")
    python, python_faces = remesh_geometry(vertices=vertices, faces=faces, iterations=iterations,
                                           scalar_storage="python")
    np.testing.assert_array_equal(default, numpy)
    np.testing.assert_array_equal(default_faces, numpy_faces)
    np.testing.assert_array_equal(python, default)
    np.testing.assert_array_equal(python_faces, default_faces)
    np.testing.assert_array_equal(vertices, copy_vertices)
    np.testing.assert_array_equal(faces, copy_faces)
    if iterations == 3:
        assert hashlib.sha256(python.astype("<f4").tobytes()+python_faces.astype("<i4").tobytes()).hexdigest() == (
            "5657a57d7a43d9fe35a42b251723a54856945acfaa23daf5ccaab53d95882b40")


def test_invalid_storage_fails_before_file_read_or_output(tmp_path):
    output = tmp_path / "output.orig"
    with pytest.raises(ValueError, match="scalar_storage"):
        remesh_surface(input_path=tmp_path/"missing.orig.premesh", output_path=output,
                       iterations=3, scalar_storage="float32")
    assert not output.exists()
