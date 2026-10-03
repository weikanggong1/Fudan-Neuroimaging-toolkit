"""Exact pair aggregation, space declarations and sparse ROI preservation."""

import csv

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.connectome.assignment import assign_endpoint_labels, build_connectomes
from fnit.connectome.paired_assignment import build_pair_connectomes
from fnit.connectome.template_inputs import (
    TemplatePair, TemplateSpec, prepare_template, template_dependency_paths,
)


DEVICES = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]


def _volume(tmp_path, name, data, affine=None):
    path = tmp_path / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.int32), np.eye(4) if affine is None else affine), path)
    return path


def _prepare(spec, **kwargs):
    return prepare_template(spec, subject_dir=None, dwi_shape=(5, 3, 3),
                            dwi_affine=np.eye(4), device="cpu", **kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_nonoverlapping_pairs_accept_reverse_and_preserve_absent_nodes(device):
    a = torch.zeros((7, 1, 1), dtype=torch.int32, device=device)
    b = torch.zeros_like(a)
    a[0, 0, 0], a[1, 0, 0] = 1, 2
    b[4, 0, 0], b[6, 0, 0] = 1, 2
    endpoints = torch.tensor([[[0, 0, 0], [4, 0, 0]],
                              [[4, 0, 0], [0, 0, 0]],
                              [[1, 0, 0], [6, 0, 0]],
                              [[0, 0, 0], [1, 0, 0]],
                              [[30, 0, 0], [4, 0, 0]]], dtype=torch.float32, device=device)
    result = build_pair_connectomes(endpoints, a, torch.eye(4, device=device),
                                    b, torch.eye(4, device=device), radius=.1,
                                    first_node_count=3, second_node_count=4,
                                    weights=torch.tensor([1, 3, 2, 7, 8], device=device),
                                    lengths=torch.tensor([10, 20, 30, 40, 50], device=device),
                                    fa=torch.tensor([.2, .8, .4, .7, .9], device=device), batch_size=1)
    assert result["count"].shape == (3, 4)
    assert result["count"][0, 0] == 2 and result["count"][1, 1] == 1
    assert result["count"].sum() == 3
    assert result["sift2_fbc"][0, 0] == 4
    assert result["mean_length"][0, 0] == 17.5
    torch.testing.assert_close(result["mean_fa"][0, 0], torch.tensor(.65, device=device))
    assert not result["count"][2].any() and not result["count"][:, 3].any()


@pytest.mark.parametrize("device", DEVICES)
def test_overlap_counts_same_cell_once_and_distinct_cells_once_each(device):
    a = torch.tensor([1, 1, 2], device=device, dtype=torch.int32).reshape(3, 1, 1)
    b = torch.tensor([1, 1, 2], device=device, dtype=torch.int32).reshape(3, 1, 1)
    endpoints = torch.tensor([[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]]],
                             device=device, dtype=torch.float32)
    result = build_pair_connectomes(endpoints, a, torch.eye(4, device=device),
                                    b, torch.eye(4, device=device), radius=.1)
    assert result["count"].tolist() == [[1, 1], [1, 0]]
    reverse = build_pair_connectomes(endpoints.flip(1), a, torch.eye(4, device=device),
                                     b, torch.eye(4, device=device), radius=.1)
    assert torch.equal(result["count"], reverse["count"])


@pytest.mark.parametrize("device", DEVICES)
def test_identity_has_exact_existing_square_outputs_including_diagonal(device):
    atlas = torch.tensor([1, 0, 3], device=device, dtype=torch.int32).reshape(3, 1, 1)
    endpoints = torch.tensor([[[0, 0, 0], [2, 0, 0]], [[2, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0]]], device=device, dtype=torch.float32)
    kwargs = dict(weights=torch.tensor([1, 3, 2], device=device),
                  lengths=torch.tensor([10, 20, 30], device=device),
                  fa=torch.tensor([.2, .8, .4], device=device), radius=.1, batch_size=1)
    actual = build_pair_connectomes(endpoints, atlas, torch.eye(4, device=device),
                                    atlas.clone(), torch.eye(4, device=device),
                                    first_node_count=4, second_node_count=4,
                                    same_template=True, **kwargs)
    expected = build_connectomes(endpoints, atlas, torch.eye(4, device=device), node_count=4, **kwargs)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert actual["count"][0, 0] == 1


def test_different_grids_strict_radius_and_axis_swap():
    a = torch.tensor([1, 2], dtype=torch.int32).reshape(2, 1, 1)
    b = torch.tensor([1, 3], dtype=torch.int32).reshape(2, 1, 1)
    a_affine, b_affine = torch.eye(4), torch.eye(4)
    b_affine[0, 3] = 10
    endpoints = torch.tensor([[[0, 0, 0], [10, 0, 0]], [[11, 0, 0], [1, 0, 0]],
                              [[-1, 0, 0], [10, 0, 0]]], dtype=torch.float32)
    actual = build_pair_connectomes(endpoints, a, a_affine, b, b_affine,
                                    first_node_count=2, second_node_count=3, radius=1.)
    swapped = build_pair_connectomes(endpoints, b, b_affine, a, a_affine,
                                     first_node_count=3, second_node_count=2, radius=1.)
    assert torch.equal(actual["count"], swapped["count"].T)
    assert actual["count"].sum() == 2  # Exactly radius 1 is excluded.


@pytest.mark.parametrize("device", DEVICES)
def test_paired_results_match_independent_per_track_oracle(device):
    # The oracle enumerates every nonzero voxel centre, independently of the
    # production radial-offset search. Coordinates avoid distance ties.
    a = torch.zeros((9, 4, 3), device=device, dtype=torch.int32)
    b = torch.zeros_like(a)
    a[1, 1, 1], a[4, 2, 1], a[7, 1, 0] = 1, 2, 3
    b[2, 1, 1], b[4, 2, 1], b[8, 1, 0] = 1, 2, 4
    gen = torch.Generator().manual_seed(481)
    endpoints = torch.rand((57, 2, 3), generator=gen) * torch.tensor([9., 4., 3.])
    endpoints = endpoints.to(device)
    weights = (torch.arange(57, device=device) % 5 + 1).float()
    lengths = torch.arange(57, device=device).float() + 1
    fa = (torch.arange(57, device=device) % 9).float() / 10
    affine = torch.tensor([[1.1, .2, 0, 0], [0, 1.3, .1, 0], [0, 0, 1.4, 0], [0, 0, 0, 1]], device=device)
    expected_count = np.zeros((4, 5), dtype=np.int64)
    expected_weights = np.zeros((4, 5), dtype=np.float64)
    expected_length = np.zeros((4, 5), dtype=np.float64)
    expected_fa = np.zeros((4, 5), dtype=np.float64)
    endpoint_np = endpoints.cpu().numpy()
    affine_np = affine.cpu().numpy()

    def nearest(atlas, point):
        indices = np.argwhere(atlas.cpu().numpy() > 0)
        world = indices @ affine_np[:3, :3].T + affine_np[:3, 3]
        distance = ((world - point) ** 2).sum(axis=1)
        k = int(distance.argmin())
        return int(atlas[tuple(indices[k])]) if distance[k] < 1.7 ** 2 else 0

    for i, (p0, p1) in enumerate(endpoint_np):
        cells = {(nearest(a, p0), nearest(b, p1)), (nearest(a, p1), nearest(b, p0))}
        for row, col in cells:
            if row and col:
                row, col = row - 1, col - 1
                weight = float(weights[i])
                expected_count[row, col] += 1
                expected_weights[row, col] += weight
                expected_length[row, col] += weight * float(lengths[i])
                expected_fa[row, col] += weight * float(fa[i])
    for batch in (1, 7, 1024):
        actual = build_pair_connectomes(endpoints, a, affine, b, affine,
                                        first_node_count=4, second_node_count=5,
                                        weights=weights, lengths=lengths, fa=fa,
                                        radius=1.7, batch_size=batch)
        assert np.array_equal(actual["count"].cpu().numpy(), expected_count)
        assert np.array_equal(actual["sift2_fbc"].cpu().numpy(), expected_weights.astype(np.float32))
        denominator = np.maximum(expected_weights, np.finfo(np.float64).tiny)
        assert np.array_equal(actual["mean_length"].cpu().numpy(), (expected_length / denominator).astype(np.float32))
        assert np.array_equal(actual["mean_fa"].cpu().numpy(), (expected_fa / denominator).astype(np.float32))


def test_sparse_volume_and_declared_absent_nodes(tmp_path):
    data = np.zeros((5, 3, 3), dtype=np.int32)
    data[0, 0, 0], data[3, 0, 0] = 1001, 2002
    volume = _volume(tmp_path, "sparse", data)
    nodes = tmp_path / "nodes.tsv"
    nodes.write_text("index\toriginal_label\themisphere\tname\n1\t2002\tR\tright\n2\t1001\tL\tleft\n3\t9009\t\tabsent\n")
    prepared = _prepare(TemplateSpec("sparse", "volume", "dwi", volume_path=volume, nodes_tsv=nodes))
    assert prepared.labels[0, 0, 0] == 2 and prepared.labels[3, 0, 0] == 1
    assert len(prepared.nodes) == 3 and prepared.nodes[2].original_label == 9009
    output = tmp_path / "written.tsv"
    prepared.write_nodes(output)
    with output.open() as stream:
        assert len(list(csv.DictReader(stream, delimiter="\t"))) == 3


def test_t1_transform_direction_and_dwi_affine(tmp_path):
    data = np.zeros((5, 3, 3), dtype=np.int32)
    data[2, 0, 0] = 99
    volume = _volume(tmp_path, "t1", data)
    pull = np.eye(4)
    pull[0, 3] = 1  # DWI voxel 1 samples T1 voxel 2.
    prepared = _prepare(TemplateSpec("t1", "volume", "t1", volume_path=volume), dwi_to_t1_world=pull)
    assert prepared.labels[1, 0, 0] == 1 and prepared.labels[2, 0, 0] == 0
    assert torch.equal(prepared.affine, torch.eye(4, dtype=torch.float64))


def _subject(tmp_path):
    root = tmp_path / "subject"
    for folder in ("mri", "surf", "label"):
        (root / folder).mkdir(parents=True)
    base = nib.MGHImage(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    nib.save(base, root / "mri/brain.mgz")
    nib.save(nib.MGHImage(np.zeros((5, 5, 5), dtype=np.int32), np.eye(4)), root / "mri/aparc+aseg.mgz")
    ribbon = np.zeros((5, 5, 5), dtype=np.int32)
    ribbon[1, 1, 1], ribbon[2, 1, 1] = 3, 3
    ribbon[1, 2, 1], ribbon[2, 2, 1] = 42, 42
    ribbon_image = nib.MGHImage(ribbon, np.eye(4))
    nib.save(ribbon_image, root / "mri/ribbon.mgz")
    tkr = ribbon_image.header.get_vox2ras_tkr()
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    for hemi, y in (("lh", 1), ("rh", 2)):
        ijk = np.array([[1, y, 1], [2, y, 1], [1, y, 2]])
        vertices = ijk @ tkr[:3, :3].T + tkr[:3, 3]
        for kind in ("white", "pial"):
            nib.freesurfer.write_geometry(root / f"surf/{hemi}.{kind}", vertices, faces)
        colors = np.array([[0, 0, 0, 0], [100, 50, 30, 0], [20, 150, 50, 0], [40, 60, 120, 0]], dtype=np.int32)
        nib.freesurfer.write_annot(root / f"label/{hemi}.test.annot", np.array([1, 2, 1], dtype=np.int32), colors,
                                   [b"unknown", b"first", b"second", b"declared_absent"])
    return root


def test_native_surface_preserves_full_lut_and_hemisphere_ids(tmp_path):
    root = _subject(tmp_path)
    spec = TemplateSpec("surface", "surface", "native", left_path=root / "label/lh.test.annot",
                        right_path=root / "label/rh.test.annot")
    prepared = prepare_template(spec, subject_dir=root, dwi_shape=(5, 5, 5),
                                dwi_affine=np.eye(4), dwi_to_t1_world=np.eye(4), device="cpu")
    assert len(prepared.nodes) == 6
    assert prepared.labels[1, 1, 1] == 1 and prepared.labels[2, 1, 1] == 2
    assert prepared.labels[1, 2, 1] == 4 and prepared.labels[2, 2, 1] == 5
    assert [node.hemisphere for node in prepared.nodes] == ["L"] * 3 + ["R"] * 3
    assert not (prepared.labels == 3).any() and not (prepared.labels == 6).any()
    assert root / "mri/ribbon.mgz" in template_dependency_paths(spec, root)


def test_pair_dict_contract_and_missing_mni_transform(tmp_path):
    volume = _volume(tmp_path, "mni", np.ones((5, 3, 3)))
    pair = TemplatePair("pair", dict(name="a", kind="volume", space="dwi", volume_path=str(volume)),
                        dict(name="b", kind="volume", space="mni", volume_path=str(volume)))
    assert isinstance(pair.first, TemplateSpec)
    with pytest.raises(ValueError, match="mni_to_t1_transform"):
        _prepare(pair.second, dwi_to_t1_world=np.eye(4))


def test_node_declaration_rejects_unknown_roi_and_fractional_volume(tmp_path):
    volume = _volume(tmp_path, "volume", np.ones((5, 3, 3)) * 2)
    nodes = tmp_path / "nodes.tsv"
    nodes.write_text("original_label\tname\n1\tonly_one\n")
    with pytest.raises(ValueError, match="absent from its node table"):
        _prepare(TemplateSpec("bad", "volume", "dwi", volume_path=volume, nodes_tsv=nodes))
    fractional = tmp_path / "fractional.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 3, 3)) * 1.5, np.eye(4)), fractional)
    with pytest.raises(ValueError, match="finite integer"):
        _prepare(TemplateSpec("bad", "volume", "dwi", volume_path=fractional))


@pytest.mark.parametrize("kwargs", [dict(name="../unsafe", kind="volume", space="dwi", volume_path="a"),
                                    dict(name="x", kind="volume", space="native", volume_path="a"),
                                    dict(name="x", kind="surface", space="fsaverage", left_path="a", right_path="b"),
                                    dict(name="x", kind="surface", space="native", left_path="a")])
def test_invalid_specs_fail_before_any_pipeline_work(kwargs):
    with pytest.raises(ValueError):
        TemplateSpec(**kwargs)


def test_identity_cannot_be_forced_for_different_labels():
    with pytest.raises(ValueError, match="same_template requires"):
        build_pair_connectomes(torch.zeros((0, 2, 3)), torch.ones((1, 1, 1), dtype=torch.int32),
                                torch.eye(4), torch.full((1, 1, 1), 2, dtype=torch.int32), torch.eye(4),
                                first_node_count=2, second_node_count=2, same_template=True)


def test_public_endpoint_rule_keeps_ties_and_strict_boundary():
    atlas = torch.tensor([1, 0, 2], dtype=torch.int32).reshape(3, 1, 1)
    assigned = assign_endpoint_labels(torch.tensor([[1., 0, 0], [-1., 0, 0], [10., 0, 0]]),
                                      atlas, torch.eye(4), radius=1.001, batch_size=1)
    assert assigned.tolist() == [1, 1, 0]
    assert assign_endpoint_labels(torch.empty((0, 3)), atlas, torch.eye(4)).shape == (0,)


def test_native_label_gifti_preserves_sparse_lut(tmp_path):
    root = _subject(tmp_path)
    paths = []
    for hemi in ("lh", "rh"):
        table = nib.gifti.GiftiLabelTable()
        for key, name in ((0, "background"), (10, "first"), (200, "second"), (999, "absent")):
            label = nib.gifti.GiftiLabel(key=key, red=.1, green=.2, blue=.3, alpha=1.)
            label.label = name
            table.labels.append(label)
        image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            np.array([10, 200, 10], dtype=np.int32), intent="NIFTI_INTENT_LABEL")], labeltable=table)
        path = root / f"label/{hemi}.test.label.gii"
        nib.save(image, path)
        paths.append(path)
    spec = TemplateSpec("gifti", "surface", "native", left_path=paths[0], right_path=paths[1])
    prepared = prepare_template(spec, subject_dir=root, dwi_shape=(5, 5, 5),
                                dwi_affine=np.eye(4), dwi_to_t1_world=np.eye(4), device="cpu")
    assert len(prepared.nodes) == 6
    assert prepared.labels[2, 1, 1] == 2 and prepared.labels[2, 2, 1] == 5
    assert [node.original_label for node in prepared.nodes] == [10, 200, 999] * 2


def test_fsaverage_sphere_mapping_uses_actual_native_vertex_order(tmp_path):
    root = _subject(tmp_path)
    average = tmp_path / "fsaverage"
    (average / "surf").mkdir(parents=True)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    source_sphere = np.array([[100., 0, 0], [0, 100., 0], [0, 0, 100.]])
    for hemi in ("lh", "rh"):
        nib.freesurfer.write_geometry(average / f"surf/{hemi}.sphere.reg", source_sphere, faces)
        nib.freesurfer.write_geometry(root / f"surf/{hemi}.sphere.reg", source_sphere[[1, 0, 2]], faces)
    spec = TemplateSpec("average", "surface", "fsaverage", fsaverage_dir=average,
                        left_path=root / "label/lh.test.annot", right_path=root / "label/rh.test.annot")
    prepared = prepare_template(spec, subject_dir=root, dwi_shape=(5, 5, 5),
                                dwi_affine=np.eye(4), dwi_to_t1_world=np.eye(4), device="cpu")
    assert prepared.labels[1, 1, 1] == 2 and prepared.labels[2, 1, 1] == 1
    assert prepared.labels[1, 2, 1] == 5 and prepared.labels[2, 2, 1] == 4
    assert average / "surf/lh.sphere.reg" in template_dependency_paths(spec, root)


def test_existing_synthmorph_dense_pull_direction_and_target_geometry(tmp_path):
    from fnit._transforms import DenseWarp

    data = np.zeros((5, 3, 3), dtype=np.int32)
    data[2, 1, 1] = 101
    source = _volume(tmp_path, "mni", data)
    reference = _volume(tmp_path, "t1", np.zeros_like(data))
    warp_values = np.zeros((5, 3, 3, 3), dtype=np.float32)
    warp_values[..., 0] = 1.  # Target T1 voxel 1 pulls source MNI voxel 2.
    warp = DenseWarp(warp_values, source=nib.load(source), target=nib.load(reference))
    spec = TemplateSpec("mni", "volume", "mni", volume_path=source)
    prepared = _prepare(spec, dwi_to_t1_world=np.eye(4), t1_reference_path=reference,
                        mni_to_t1_transform=warp)
    assert prepared.labels[1, 1, 1] == 1 and prepared.labels[2, 1, 1] == 0
    shifted = _volume(tmp_path, "wrong_t1", np.zeros_like(data), np.diag([2, 2, 2, 1]))
    with pytest.raises(ValueError, match="target must match"):
        _prepare(spec, dwi_to_t1_world=np.eye(4), t1_reference_path=shifted,
                 mni_to_t1_transform=warp)
