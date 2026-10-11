"""检查零相交保留和非零相交修复结果复检。"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

pytest.importorskip("nibabel")
pytest.importorskip("scipy")
import nibabel.freesurfer as fs

from fnit.recon_all.mris_remove_intersection_python import (
    mark_intersections,
    remove_intersection_surface,
)


def tetrahedra(offset: float) -> tuple[np.ndarray, np.ndarray]:
    tetra = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return np.concatenate((tetra, tetra + offset)), np.concatenate((faces, faces + 4))


def test_collision_marks_match_native_positive_and_negative_examples() -> None:
    vertices, faces = tetrahedra(0.3)
    marks, nfaces = mark_intersections(vertices, faces)
    assert nfaces == 4
    assert marks.tolist() == [False, True, True, True, True, True, True, True]

    vertices, faces = tetrahedra(3.0)
    marks, nfaces = mark_intersections(vertices, faces)
    assert nfaces == 0
    assert not marks.any()


def test_shared_vertex_is_not_an_intersection() -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    marks, nfaces = mark_intersections(vertices, faces)
    assert nfaces == 0
    assert not marks.any()


def test_coplanar_triangles_with_overlapping_boxes() -> None:
    first = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    separated = np.array([[0.8, 0.8, 0], [1.8, 0.8, 0], [0.8, 1.8, 0]], dtype=np.float32)
    crossing = np.array([[0.2, 0.2, 0], [1.2, 0.2, 0], [0.2, 1.2, 0]], dtype=np.float32)
    assert mark_intersections(np.concatenate((first, separated)), faces)[1] == 0
    assert mark_intersections(np.concatenate((first, crossing)), faces)[1] == 2


def test_zero_branch_preserves_surface_file(tmp_path) -> None:
    vertices, faces = tetrahedra(3.0)
    source, output = tmp_path / "input", tmp_path / "output"
    fs.write_geometry(str(source), vertices, faces)
    assert remove_intersection_surface(source, output) == (0, 0)
    assert output.read_bytes() == source.read_bytes()
    original_bytes = source.read_bytes()
    assert remove_intersection_surface(source, source) == (0, 0)
    assert source.read_bytes() == original_bytes


def test_positive_branch_rechecks_repaired_surface(tmp_path) -> None:
    vertices, faces = tetrahedra(0.3)
    source, output = tmp_path / "input", tmp_path / "output"
    fs.write_geometry(str(source), vertices, faces)
    binary = tmp_path / "mris_remove_intersection"
    binary.write_text("binary")

    corrected, corrected_faces = tetrahedra(3.0)
    np.testing.assert_array_equal(corrected_faces, faces)
    diagnostics = dict(intersecting_faces_before=4, marked_vertices=7)
    # 文件调度契约：算法本身另有真实非零交回归。本测试不执行旧binary。
    with patch("fnit.recon_all.place_surface_final_cleanup.repair_intersections",
               return_value=(corrected, diagnostics)) as repair:
        assert remove_intersection_surface(source, output, binary=binary,
                                           assets_dir=tmp_path) == (4, 7)
        assert repair.call_args.kwargs["marking_backend"] == "source_numba"
    assert mark_intersections(*fs.read_geometry(str(output)))[1] == 0


def test_residual_repair_is_rejected_before_output(tmp_path) -> None:
    vertices, faces = tetrahedra(0.3)
    source, output = tmp_path / "input", tmp_path / "output"
    fs.write_geometry(str(source), vertices, faces)
    original = source.read_bytes()
    with patch("fnit.recon_all.place_surface_final_cleanup.repair_intersections",
               return_value=(vertices, dict(intersecting_faces_before=4, marked_vertices=7))):
        with pytest.raises(RuntimeError, match="intersecting faces"):
            remove_intersection_surface(source, output)
    assert not output.exists()
    assert source.read_bytes() == original
