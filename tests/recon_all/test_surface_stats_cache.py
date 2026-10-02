"""缓存身份、no-th3 语义及汇总回归；人工小网格只用于单元测试。"""

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pytest
import torch

from fnit.recon_all.surface_area_gpu import vertex_area
from fnit.recon_all.surface_roi_curvature_gpu import principal_curvatures
from fnit.recon_all.surface_roi_gpu import vertex_th3_volume
from fnit.recon_all.surface_stats_cache import SurfaceStatsCache


@pytest.fixture
def files(tmp_path: Path):
    xyz = np.asarray(((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                      (0, -1, 0), (0, 0, 1), (0, 0, -1)), dtype=np.float32)
    faces = np.asarray(((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
                        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)), dtype=np.int32)
    paths = {name: tmp_path / ("lh." + name) for name in
             ("white", "white.preaparc", "pial", "area", "area.pial", "thickness", "aparc.annot", "DKT.annot", "cortex.label")}
    pial = xyz * np.asarray((1.3, 1.5, 1.8), dtype=np.float32)
    for name, vertices in (("white", xyz), ("white.preaparc", xyz * 0.9), ("pial", pial)):
        fsio.write_geometry(str(paths[name]), vertices, faces)
    thickness = np.asarray((1.5, 1.8, 2, 2.2, 1.6, 2.4), dtype=np.float32)
    fsio.write_morph_data(str(paths["thickness"]), thickness)
    fsio.write_morph_data(str(paths["area"]), vertex_area(xyz, faces, device="cpu"))
    fsio.write_morph_data(str(paths["area.pial"]), vertex_area(pial, faces, device="cpu"))
    labels = np.asarray((1, 1, 2, 2, 2, 1), dtype=np.int32)
    colors = np.asarray(((0, 0, 0, 0), (1, 2, 3, 0), (4, 5, 6, 0)), dtype=np.int32)
    for atlas, region in (("aparc.annot", labels), ("DKT.annot", labels[::-1])):
        fsio.write_annot(str(paths[atlas]), region, colors, [b"unknown", b"A", b"B"])
    paths["cortex.label"].write_text("#!ascii label\n6\n" + "".join(
        f"{i} {x} {y} {z} 0\n" for i, (x, y, z) in enumerate(xyz)))
    return paths, xyz, pial, faces, thickness, labels


def test_roi_summary_matches_ordered_scalar_reductions(files):
    paths, xyz, _, faces, thickness, labels = files
    with SurfaceStatsCache(device="cpu") as cache:
        basic, volumes = cache.roi_base(paths["white"], paths["aparc.annot"], paths["thickness"],
                                       white=paths["white"], pial=paths["pial"])
        area = vertex_area(xyz, faces, device="cpu").astype(np.float64)
        volume = cache.no_th3_volume(paths["white"], paths["pial"], paths["thickness"])
        for index, name in ((1, "A"), (2, "B")):
            selected = labels == index
            values = torch.as_tensor(thickness[selected].astype(np.float64))
            expected = (int(selected.sum()), float(torch.as_tensor(area[selected]).sum()),
                        float(values.mean()), float(values.std(unbiased=False)))
            assert basic[name] == expected
            assert volumes[name] == float(volume[torch.as_tensor(selected)].sum())
        cache.roi_base(paths["white"], paths["DKT.annot"], paths["thickness"],
                       white=paths["white"], pial=paths["pial"])
        assert cache.counters["geometry_reads"] == 2
        assert cache.counters["vertex_area_computations"] == 1
        assert cache.counters["no_th3_volume_computations"] == 1
        assert cache.counters["morph_reads"] == 1
        assert cache.counters["roi_summary_transfers"] == 2
    assert not cache._geometries
    assert not cache._volumes


def test_no_th3_formula_does_not_use_th3_vertex_volume(files):
    paths, xyz, pial, faces, thickness, _ = files
    w, p = (torch.as_tensor(v) for v in (xyz, pial))
    tri = torch.as_tensor(faces.astype(np.int64))
    face_area = []
    for vertices in (w, p):
        a, b, c = (vertices[tri[:, i]] for i in range(3))
        face_area.append(torch.linalg.vector_norm(torch.cross(b - a, c - a, dim=1), dim=1) * 0.5)
    share = torch.as_tensor(thickness)[tri].to(torch.float64).mean(1) * (
        face_area[0].to(torch.float64) + face_area[1].to(torch.float64)) / 6.0
    expected = torch.zeros(len(xyz), dtype=torch.float64)
    for corner in range(3):
        expected.index_add_(0, tri[:, corner], share)
    with SurfaceStatsCache(device="cpu") as cache:
        actual = cache.no_th3_volume(paths["white"], paths["pial"], paths["thickness"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        th3 = vertex_th3_volume(paths["white"], paths["pial"], paths["cortex.label"], device="cpu")
        assert not np.allclose(actual.numpy(), th3)


def test_surface_identity_includes_path_and_overwritten_version(files):
    paths, xyz, _, faces, _, _ = files
    with SurfaceStatsCache(device="cpu") as cache:
        white = cache.geometry(paths["white"])
        pre = cache.geometry(paths["white.preaparc"])
        pial = cache.geometry(paths["pial"])
        assert white is not pre and white is not pial and pre is not pial
        assert np.array_equal(white.faces, pre.faces)
        assert not np.array_equal(white.vertices, pre.vertices)
        first = cache.principal(white)
        second = cache.principal(white)
        assert first is second
        expected = principal_curvatures(xyz, faces, device="cpu")
        for before, after in zip(expected, first):
            np.testing.assert_array_equal(before, after)
        for geometry in (pre, pial):
            cache.principal(geometry)
        fsio.write_geometry(str(paths["white"]), xyz * 1.1, faces)
        modified = cache.geometry(paths["white"])
        assert modified is not white
        cache.principal(modified)
        assert cache.counters["geometry_reads"] == 4
        assert cache.counters["principal_curvature_computations"] == 4


def test_cortex_mask_is_not_mutated_and_topology_is_checked(files):
    paths, xyz, pial, faces, _, _ = files
    with SurfaceStatsCache(device="cpu") as cache:
        selected = cache.cortex(paths["cortex.label"], len(xyz))
        assert selected.all()
        for atlas in ("aparc.annot", "DKT.annot"):
            columns = cache.curvature_columns(paths["white"], paths["area"], paths[atlas], paths["cortex.label"])
            assert set(columns) == {"A", "B"}
        assert selected.all()
        assert cache.counters["principal_curvature_computations"] == 1
        fsio.write_geometry(str(paths["pial"]), pial, faces[::-1])
        with pytest.raises(ValueError, match="ordered topology"):
            cache.no_th3_volume(paths["white"], paths["pial"], paths["thickness"])


def test_cache_rejects_incorrect_device(files):
    cache = SurfaceStatsCache(device="cpu")
    with pytest.raises(ValueError, match="device"):
        cache.check_device("meta")
