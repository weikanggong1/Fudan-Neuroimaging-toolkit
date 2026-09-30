import csv
import gzip
import struct

import nibabel as nib
import numpy as np
import pytest

from fnit.bwas import plot_bwas_connectivity
import fnit.bwas.visualize as visualization
from fnit.bwas.visualize import _result_files, _significant_clusters, _top_edges


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


def test_streaming_top_edges_filter_significance_and_rank(tmp_path):
    _, _, edges, clusters = _example(tmp_path)
    significant = _significant_clusters(clusters, 0.05)
    assert significant == {1}
    selected = _top_edges(edges, significant, top_k=2, min_abs_z=5.5)
    assert [entry[1] for entry in selected] == [-7.0, 6.0]
    assert all(entry[2] == 1 for entry in selected)
    assert _top_edges(edges, set(), top_k=2, min_abs_z=None) == []


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
