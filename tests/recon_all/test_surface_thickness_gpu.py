import importlib.util
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pytest
import torch

from fnit.recon_all import surface_thickness_gpu as stage
from fnit.recon_all.surface_thickness_gpu import thickness_map

_dense_path = Path(__file__).resolve().parents[2] / (
    "validation/recon_all/python_gpu_port/reference_surface_thickness_dense_3d9856c.py")
_dense_spec = importlib.util.spec_from_file_location("frozen_dense_thickness_test", _dense_path)
dense_stage = importlib.util.module_from_spec(_dense_spec)
_dense_spec.loader.exec_module(dense_stage)
dense_thickness_map = dense_stage.thickness_map


def test_compiled_reachability_keeps_twenty_hop_boundary():
    # 0→20 可达；0→21 超过限制，不能采用其更短距离。
    faces = np.array([[v, v + 1, v + 1] for v in range(22)], dtype=np.int32)
    graph = stage._adjacency_csr(faces, 23)
    distances = stage._reachable_distances(
        0, np.array([2.0], dtype=np.float32), graph.indptr, graph.indices,
        np.array([0, 2]), np.array([21, 20], dtype=np.int32),
        np.array([True, True]), np.array([0.25, 0.5], dtype=np.float32),
        np.array([0, 1]), np.array([0], dtype=np.int32),
        np.array([True]), np.array([1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(distances, [[0.5, 1.0]])


def test_compiled_reachability_checks_all_equal_distance_candidates():
    graph = stage._adjacency_csr(np.array([[0, 1, 2]], dtype=np.int32), 4)
    distances = stage._reachable_distances(
        0, np.array([2.0], dtype=np.float32), graph.indptr, graph.indices,
        np.array([0, 2]), np.array([3, 1], dtype=np.int32),
        np.array([True, True]), np.array([0.5, 0.5], dtype=np.float32),
        np.array([0, 0]), np.array([], dtype=np.int32),
        np.array([], dtype=bool), np.array([], dtype=np.float32),
    )
    np.testing.assert_array_equal(distances, [[0.5, 2.0]])


def test_reachability_workspace_does_not_leak_previous_vertices():
    graph = stage._adjacency_csr(np.array([[0, 1, 2]], dtype=np.int32), 4)
    seen, queue = np.zeros(4, dtype=np.int32), np.empty(4, dtype=np.int32)
    empty = (np.array([0, 0]), np.array([], dtype=np.int32),
             np.array([], dtype=bool), np.array([], dtype=np.float32))
    candidate = (np.array([0, 1]), np.array([1], dtype=np.int32),
                 np.array([True]), np.array([0.5], dtype=np.float32))
    stage._reachable_distances(0, np.array([2.0], dtype=np.float32),
                               graph.indptr, graph.indices, *candidate, *empty, seen, queue)
    result = stage._reachable_distances(3, np.array([2.0], dtype=np.float32),
                                        graph.indptr, graph.indices, *candidate, *empty, seen, queue)
    np.testing.assert_array_equal(result, [[2.0, 2.0]])


def test_indexed_search_includes_candidates_beyond_256(tmp_path, monkeypatch):
    from scipy.sparse import csr_matrix

    white = np.zeros((259, 3), dtype=np.float32)
    white[1:, 0] = np.arange(1001, 1259)
    pial = np.zeros_like(white)
    pial[0, 2], pial[1:258, 2], pial[258, 2] = 4, -.5, 1
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    white_file, pial_file, output = (tmp_path / name for name in
                                     ("white", "pial", "thickness"))
    fs.write_geometry(str(white_file), white, faces)
    fs.write_geometry(str(pial_file), pial, faces)
    monkeypatch.setattr(stage, "_normals", lambda vertices, _: torch.tensor(
        [0., 0., 1.], device=vertices.device).expand_as(vertices))
    monkeypatch.setattr(stage, "_adjacency_csr", lambda _, n: csr_matrix(
        np.ones((n, n), dtype=bool)))
    report = stage.thickness_map_indexed(white_file, pial_file, output, device="cpu")
    np.testing.assert_equal(fs.read_morph_data(str(output))[0], np.float32(2.5))
    assert report["candidate_pairs"] >= 259


def test_indexed_cpu_matches_existing_folded_surface(tmp_path):
    grid = np.arange(16, dtype=np.float32) * 0.3
    x, y = np.meshgrid(grid, grid)
    white = np.column_stack((x.ravel(), y.ravel(), np.sin(x.ravel()))).astype(np.float32)
    pial = white.copy()
    pial[:, 2] += 1.5 + 0.25 * np.cos(y.ravel())
    faces = []
    for row in range(15):
        for column in range(15):
            v = row * 16 + column
            faces.extend(((v, v + 1, v + 16), (v + 1, v + 17, v + 16)))
    white_file, pial_file = tmp_path / "white", tmp_path / "pial"
    fs.write_geometry(str(white_file), white, np.asarray(faces))
    fs.write_geometry(str(pial_file), pial, np.asarray(faces))
    dense_thickness_map(white_file, pial_file, tmp_path / "old", device="cpu")
    stage.thickness_map_indexed(white_file, pial_file, tmp_path / "new", device="cpu")
    np.testing.assert_allclose(fs.read_morph_data(str(tmp_path / "old")),
                               fs.read_morph_data(str(tmp_path / "new")),
                               rtol=0, atol=1e-6)


def test_clip_mean_preserves_existing_float32_conversion():
    distances = np.array([[1.5, 2.25], [5., 1.1234567], [5.5, 1.1234567],
                          [7., 8.]], dtype=np.float32)
    old = np.asarray([(min(a, 5.) + min(b, 5.)) / 2 for a, b in distances],
                      dtype=np.float32)
    np.testing.assert_array_equal(stage._clip_mean(distances), old)


def test_parallel_surfaces_have_unit_thickness(tmp_path):
    white = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    pial = white.copy()
    pial[:, 2] = 1
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    white_file, pial_file, output = (tmp_path / name for name in
                                     ("white", "pial", "thickness"))
    fs.write_geometry(str(white_file), white, faces)
    fs.write_geometry(str(pial_file), pial, faces)
    report = thickness_map(white_file, pial_file, output, device="cpu")
    np.testing.assert_allclose(fs.read_morph_data(str(output)), 1, atol=1e-6)
    assert report["vertices"] == 4


def test_rejects_different_white_pial_topology(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    white_file, pial_file = tmp_path / "white", tmp_path / "pial"
    fs.write_geometry(str(white_file), vertices, np.array([[0, 1, 2]], dtype=np.int32))
    fs.write_geometry(str(pial_file), vertices, np.array([[0, 2, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="identical topology"):
        thickness_map(white_file, pial_file, tmp_path / "thickness", device="cpu")


def test_expands_search_when_256_nearest_are_rejected(tmp_path, monkeypatch):
    white = np.zeros((259, 3), dtype=np.float32)
    white[1:, 0] = np.arange(1001, 1259)
    pial = np.zeros_like(white)
    pial[0, 2] = 4
    pial[1:258, 2] = -.5
    pial[258, 2] = 1
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    white_file, pial_file, output = (tmp_path / name for name in
                                     ("white", "pial", "thickness"))
    fs.write_geometry(str(white_file), white, faces)
    fs.write_geometry(str(pial_file), pial, faces)
    monkeypatch.setattr(dense_stage, "_normals", lambda vertices, _: torch.tensor(
        [0., 0., 1.], device=vertices.device).expand_as(vertices))
    monkeypatch.setattr(dense_stage, "_adjacency", lambda _, n: [list(range(n)) for _ in range(n)])
    report = dense_thickness_map(white_file, pial_file, output, device="cpu")
    assert fs.read_morph_data(str(output))[0] == pytest.approx(2.5)
    assert report["expanded_searches"] >= 1
