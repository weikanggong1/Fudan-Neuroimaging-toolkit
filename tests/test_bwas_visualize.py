import csv
import gzip
import struct

import nibabel as nib
import numpy as np
import pytest

from fnit.bwas import plot_bwas_connectivity
import fnit.bwas.visualize as visualization
from fnit.bwas.visualize import (_brainnet_surface, _cluster_bundles, _connection_paths,
                                 _check_surface_projection, _mask_surface,
                                 _result_files, _significant_clusters, _top_edges)


def _example(tmp_path):
    result_root = tmp_path / "bwas"
    folder = result_root / "group" / "func"
    folder.mkdir(parents=True)
    prefix = "task-rest_space-MNI152NLin6Asym_res-2"
    clusters = folder / f"{prefix}_desc-BWASclusters_stat.tsv"
    with clusters.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["cluster", "edges", "p_fwer", "p_uncorrected", "max_z",
                         "region1_voxels", "region2_voxels"])
        writer.writerows([(1, 3, 0.01, 0.001, 7, 2, 2),
                          (2, 1, 0.20, 0.10, 9, 1, 1)])
    edges = folder / f"{prefix}_desc-BWASedges_relmat.tsv.gz"
    with gzip.open(edges, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["voxel1_i", "voxel1_j", "voxel1_k", "voxel2_i",
                         "voxel2_j", "voxel2_k", "z", "cluster"])
        writer.writerows([
            (1, 1, 1, 5, 5, 3, 5.0, 1),
            (2, 2, 2, 4, 4, 3, -7.0, 1),
            (3, 3, 1, 3, 4, 3, 6.0, 1),
            (2, 2, 1, 4, 4, 2, 9.0, 2),
        ])
    mask = np.zeros((7, 7, 5), dtype=np.uint8)
    mask[1:6, 1:6, 1:4] = 1
    ma = np.zeros_like(mask, dtype=np.float32)
    ma[1, 1, 1], ma[5, 5, 3], ma[2, 2, 2], ma[4, 4, 3] = 1, 2, 3, 4
    affine = np.array([[2, 0, 0, -6], [0, 2, 0, -6],
                       [0, 0, 2, -4], [0, 0, 0, 1]], dtype=float)
    mask_file = tmp_path / "gray_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), mask_file)
    nib.save(nib.Nifti1Image(ma, affine),
             folder / f"{prefix}_desc-BWASMA_statmap.nii.gz")
    return result_root, mask_file, edges, clusters


def _box_surface(path, bounds):
    import pyvista as pv

    mesh = pv.Box(bounds=bounds).triangulate()
    faces = mesh.faces.reshape(-1, 4)[:, 1:] + 1
    with path.open("w") as stream:
        stream.write(f"{mesh.n_points}\n")
        np.savetxt(stream, mesh.points)
        stream.write(f"{len(faces)}\n")
        np.savetxt(stream, faces, fmt="%d")
    return path


def test_streaming_top_edges_filter_significance_and_rank(tmp_path):
    _, _, edges, clusters = _example(tmp_path)
    significant = _significant_clusters(clusters, 0.05)
    assert significant == {1}
    selected = _top_edges(edges, significant, top_k=2, min_abs_z=5.5)
    assert [entry[1] for entry in selected] == [-7.0, 6.0]
    assert all(entry[2] == 1 for entry in selected)
    assert _top_edges(edges, set(), top_k=2, min_abs_z=None) == []


def test_visual_bundling_preserves_endpoints_and_sign_groups():
    world = np.array([[[0, 0, 0], [10, 0, 0]],
                      [[0, 4, 0], [10, 4, 0]],
                      [[0, 8, 0], [10, 8, 0]]], dtype=float)
    points, lines = _connection_paths(world, np.array([1, 1, 1]),
                                      np.array([5.0, 6.0, -5.0]), 1.0)
    paths = points.reshape(3, 9, 3)
    np.testing.assert_array_equal(paths[:, 0], world[:, 0])
    np.testing.assert_array_equal(paths[:, -1], world[:, 1])
    np.testing.assert_allclose(paths[0, 4], paths[1, 4])
    np.testing.assert_allclose(paths[2, 4], [5, 8, 0])
    assert len(lines) == 3 * 10


def test_cluster_bundles_use_every_significant_edge(tmp_path):
    _, mask, edges, _ = _example(tmp_path)
    bundles = _cluster_bundles(edges, {1}, nib.load(mask).affine)
    assert len(bundles) == 2
    assert sum(bundle[1] for bundle in bundles) == 3
    assert {(bundle[1], bundle[2]) for bundle in bundles} == {(1, -7.0), (2, 5.5)}
    assert sum(len(bundle[4]) for bundle in bundles) == 3
    assert {sample[0] for bundle in bundles for sample in bundle[4]} == {
        (1, 1, 1, 5, 5, 3), (2, 2, 2, 4, 4, 3), (3, 3, 1, 3, 4, 3)}
    with pytest.raises(ValueError, match="cover every significant"):
        _cluster_bundles(edges, {1}, nib.load(mask).affine, edge_budget=1)


def test_cluster_voxel_budget_is_proportional_and_keeps_every_cluster(tmp_path):
    _, mask, edges, _ = _example(tmp_path)
    with gzip.open(edges, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["voxel1_i", "voxel1_j", "voxel1_k", "voxel2_i",
                         "voxel2_j", "voxel2_k", "z", "cluster"])
        for cluster, count in ((1, 2), (2, 10), (3, 40)):
            writer.writerows((index, 1, 1, 5, 5, 3, 5.0, cluster)
                             for index in range(count))
    bundles = _cluster_bundles(edges, {1, 2, 3}, nib.load(mask).affine, edge_budget=12)
    repeated = _cluster_bundles(edges, {1, 2, 3}, nib.load(mask).affine, edge_budget=12)
    samples = {cluster: len(sample) for cluster, _, _, _, sample in bundles}
    assert sum(samples.values()) == 12
    assert 1 <= samples[1] < samples[2] < samples[3]
    assert [bundle[4] for bundle in bundles] == [bundle[4] for bundle in repeated]


def test_brainnet_surface_uses_one_based_face_indices(tmp_path):
    path = tmp_path / "template.nv"
    path.write_text("4\n0 0 0\n2 0 0\n0 2 0\n0 0 2\n4\n"
                    "1 2 3\n1 2 4\n1 3 4\n2 3 4\n")
    vertices, faces = _brainnet_surface(path)
    assert vertices.shape == (4, 3)
    np.testing.assert_array_equal(faces[0], [0, 1, 2])
    path.write_text(path.read_text().replace("2 3 4", "2 3 5"))
    with pytest.raises(ValueError, match="face indices"):
        _brainnet_surface(path)


def test_plot_montage_reads_edges_once_and_refuses_overwrite(tmp_path, monkeypatch):
    root, mask, *_ = _example(tmp_path)
    output = tmp_path / "figures" / "bwas.png"
    original = visualization._top_edges
    calls = []

    def tracked_edges(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(visualization, "_top_edges", tracked_edges)
    assert plot_bwas_connectivity(root, mask, output, top_k=2) == output
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", output.read_bytes()[16:24]) == (2160, 1800)
    assert calls == [1]
    assert output.stat().st_size > 1000
    with pytest.raises(FileExistsError):
        plot_bwas_connectivity(root, mask, output)


@pytest.mark.parametrize("view", ["superior", "left", "right", "anterior", "oblique"])
def test_single_brain_views(tmp_path, view):
    root, mask, *_ = _example(tmp_path)
    output = tmp_path / f"{view}.png"
    assert plot_bwas_connectivity(root, mask, output, top_k=2, view=view) == output
    assert struct.unpack(">II", output.read_bytes()[16:24]) == (1440, 1440)


def test_invalid_view(tmp_path):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match="view must be montage"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png", view="bottom")


@pytest.mark.parametrize("cluster_id", [0, -1, True, 1.5])
def test_cluster_id_must_be_positive_integer(tmp_path, cluster_id):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match="cluster_id must be a positive integer"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png", cluster_id=cluster_id)


def test_cluster_id_must_pass_significance_threshold(tmp_path):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match="cluster_id is not significant"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png", cluster_id=2)


def test_selected_cluster_is_passed_to_edge_filter(tmp_path, monkeypatch):
    root, mask, *_ = _example(tmp_path)
    selected = []
    original = visualization._top_edges

    def tracked_edges(path, clusters, top_k, min_abs_z):
        selected.append(clusters)
        return original(path, clusters, top_k, min_abs_z)

    monkeypatch.setattr(visualization, "_top_edges", tracked_edges)
    plot_bwas_connectivity(root, mask, tmp_path / "cluster.png", cluster_id=1)
    assert selected == [{1}]


@pytest.mark.parametrize("view", ["six", "signed_six"])
def test_all_clusters_six_views_with_supplied_surface(tmp_path, view):
    root, mask, *_ = _example(tmp_path)
    surface_file = _box_surface(tmp_path / "template.nv", (-7, 7, -7, 7, -5, 7))
    cerebellum_file = tmp_path / "cerebellum.nv"
    cerebellum_file.write_text("4\n-2 -3 -9\n2 -3 -9\n0 1 -9\n0 -1 -5\n4\n"
                               "1 2 3\n1 2 4\n1 3 4\n2 3 4\n")
    output = tmp_path / "bundles.png"
    plot_bwas_connectivity(root, mask, output, all_clusters=True, top_k=0,
                           voxel_edge_budget=2, brain_surface_file=surface_file,
                           cerebellum_surface_file=cerebellum_file,
                           surface_opacity=0.4, colorbar_max_abs_z=8.0,
                           show_colorbar=False, view=view)
    assert struct.unpack(">II", output.read_bytes()[16:24]) == (2700, 1800)


def test_mismatched_surface_refuses_to_draw_valid_voxels_outside_outline(tmp_path):
    root, mask, *_ = _example(tmp_path)
    surface_file = _box_surface(tmp_path / "small.nv", (-1, 1, -1, 1, -1, 1))
    output = tmp_path / "wrong.png"
    with pytest.raises(ValueError, match="brain surface misses.*brain_mask_file"):
        plot_bwas_connectivity(root, mask, output, brain_surface_file=surface_file)
    assert not output.exists()


def test_mask_surface_keeps_isolated_boundary_voxel_centers():
    import pyvista as pv

    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[0, 0, 0] = mask[7, 7, 7] = True
    affine = np.diag([2., 2., 2., 1.])
    vertices, faces = _mask_surface(mask, affine)
    surface = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    world = nib.affines.apply_affine(affine, np.argwhere(mask))
    assert surface.n_open_edges == 0
    _check_surface_projection(surface, world, {
        "left": (-1, 0, 0), "superior": (0, 0, 1), "oblique": (1, -1, 0.7)})


def test_matching_brain_mask_draws_complete_outline(tmp_path):
    root, mask, *_ = _example(tmp_path)
    image = nib.load(mask)
    brain_file = tmp_path / "brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(image.shape, dtype=np.uint8), image.affine), brain_file)
    output = tmp_path / "brain.png"
    plot_bwas_connectivity(root, mask, output, brain_mask_file=brain_file,
                           all_clusters=True, bundle_strength=0.95, view="signed_six")
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("error", ["shape", "affine", "empty", "nonbinary"])
def test_brain_mask_must_be_matching_and_binary(tmp_path, error):
    root, mask, *_ = _example(tmp_path)
    image = nib.load(mask)
    shape = (8, 7, 5) if error == "shape" else image.shape
    affine = image.affine.copy()
    if error == "affine":
        affine[0, 3] += 10
    values = np.full(shape, 0 if error == "empty" else 2 if error == "nonbinary" else 1,
                     dtype=np.uint8)
    brain_file = tmp_path / "wrong_brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(values, affine), brain_file)
    with pytest.raises(ValueError, match="brain_mask_file"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png", brain_mask_file=brain_file)


def test_all_clusters_rejects_edge_filters(tmp_path):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match="cannot be combined"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png",
                               all_clusters=True, cluster_id=1)


def test_signed_six_requires_cluster_mode(tmp_path):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match="signed_six requires all_clusters"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png", view="signed_six")


@pytest.mark.parametrize("parameter,value,message", [
    ("surface_opacity", 1.1, "surface_opacity"),
    ("colorbar_max_abs_z", 0, "colorbar_max_abs_z"),
    ("voxel_edge_budget", 0, "voxel_edge_budget"),
])
def test_visual_parameter_ranges(tmp_path, parameter, value, message):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match=message):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png",
                               all_clusters=True, **{parameter: value})


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_bundle_strength_range(tmp_path, strength):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match="bundle_strength must be in"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png", bundle_strength=strength)


@pytest.mark.parametrize("top_k", [0, 1.5, float("inf"), True])
def test_top_k_must_be_positive_integer(tmp_path, top_k):
    root, mask, *_ = _example(tmp_path)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        plot_bwas_connectivity(root, mask, tmp_path / "wrong.png", top_k=top_k)


def test_no_significant_edges_still_writes_figure(tmp_path):
    root, mask, *_ = _example(tmp_path)
    output = tmp_path / "empty.png"
    assert plot_bwas_connectivity(root, mask, output, cluster_p_max=0.001) == output
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_checks_grid_and_unique_group_result(tmp_path):
    root, mask, edges, _ = _example(tmp_path)
    wrong_mask = tmp_path / "wrong_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((7, 7, 5), dtype=np.uint8),
                             np.diag([3.0, 2.0, 2.0, 1.0])), wrong_mask)
    with pytest.raises(ValueError, match="affines differ"):
        plot_bwas_connectivity(root, wrong_mask, tmp_path / "wrong.png")
    duplicate = edges.with_name("task-other" + edges.name)
    duplicate.write_bytes(edges.read_bytes())
    with pytest.raises(ValueError, match="exactly one BWAS edge file"):
        _result_files(root)
