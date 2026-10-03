"""Unilateral surface ROI templates without fictitious contralateral nodes.

These tiny MRI/surface fixtures test source validation, actual native/fsaverage
label projection and endpoint matrices on CPU. They are not a MRI benchmark.
"""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.connectome.freesurfer_subject import ConnectomeNode
from fnit.connectome.paired_assignment import build_pair_connectomes
from fnit.connectome.template_inputs import (
    TemplateSpec, _canonical_labels, preflight_template, prepare_template,
)


def _gifti(path, values, entries):
    table = nib.gifti.GiftiLabelTable()
    for key, name in entries:
        label = nib.gifti.GiftiLabel(key=key, red=.1, green=.2, blue=.3, alpha=1.)
        label.label = name
        table.labels.append(label)
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
        np.asarray(values, np.int32), intent="NIFTI_INTENT_LABEL")], labeltable=table), path)
    return path


@pytest.fixture
def subject(tmp_path):
    root = tmp_path / "subject"
    for folder in ("mri", "surf", "label"):
        (root / folder).mkdir(parents=True)
    shape = (5, 5, 5)
    brain = nib.MGHImage(np.ones(shape, np.float32), np.eye(4))
    nib.save(brain, root / "mri/brain.mgz")
    nib.save(nib.MGHImage(np.zeros(shape, np.int32), np.eye(4)), root / "mri/aparc+aseg.mgz")
    ribbon = np.zeros(shape, np.int32)
    ribbon[1:3, 1, 1], ribbon[1:3, 2, 1] = 3, 42
    image = nib.MGHImage(ribbon, np.eye(4))
    nib.save(image, root / "mri/ribbon.mgz")
    faces = np.array([[0, 1, 2]], np.int32)
    tkr = image.header.get_vox2ras_tkr()
    for hemi, y in (("lh", 1), ("rh", 2)):
        ijk = np.array([[1, y, 1], [2, y, 1], [1, y, 2]])
        vertices = ijk @ tkr[:3, :3].T + tkr[:3, 3]
        for kind in ("white", "pial", "sphere.reg"):
            nib.freesurfer.write_geometry(root / f"surf/{hemi}.{kind}", vertices, faces)
    return root


def _spec(root, *, side="L", nodes=False, space="native", other_values=(0, 0, 0)):
    paths = []
    for hemi, hemisphere in (("lh", "L"), ("rh", "R")):
        active = side == hemisphere
        values = (17, 17, 0) if active else other_values
        entries = ((0, "background"), (17, "one")) if active else ((0, "background"),)
        paths.append(_gifti(root / f"label/{hemi}.roi.label.gii", values, entries))
    table = None
    if nodes:
        table = root / "nodes.tsv"
        table.write_text(f"index\toriginal_label\themisphere\tname\n1\t17\t{side}\tone\n")
    return TemplateSpec(f"one_{side}", "surface", space, left_path=paths[0], right_path=paths[1],
                        nodes_tsv=table, fsaverage_dir=root if space == "fsaverage" else None)


def _prepare(spec, subject):
    return prepare_template(spec, subject_dir=subject, dwi_shape=(5, 5, 5),
                            dwi_affine=np.eye(4), dwi_to_t1_world=np.eye(4), device="cpu")


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("nodes", [False, True])
@pytest.mark.parametrize("space", ["native", "fsaverage"])
def test_unilateral_native_and_fsaverage_have_one_node_and_zero_other_hemisphere(subject, side, nodes, space):
    spec = _spec(subject, side=side, nodes=nodes, space=space)
    report = preflight_template(spec, subject_dir=subject)
    assert report["node_count"] == 1 and report["native_geometry"] == "checked"
    prepared = _prepare(spec, subject)
    assert len(prepared.nodes) == 1 and prepared.nodes[0].hemisphere == side
    assert prepared.labels.dtype == torch.int32
    active, inactive = (1, 2) if side == "L" else (2, 1)
    assert prepared.labels[1:3, active, 1].tolist() == [1, 1]
    assert prepared.labels[:, inactive, :].count_nonzero() == 0
    assert prepared.labels.count_nonzero() == 2


def test_unilateral_ss_and_sv_use_real_one_by_one_matrices(subject):
    left = _prepare(_spec(subject, side="L", nodes=True), subject)
    right = _prepare(_spec(subject, side="R", nodes=True), subject)
    volume_data = np.zeros((5, 5, 5), np.int32)
    volume_data[1:3, 2, 1] = 99
    volume_path = subject / "right_dwi.nii.gz"
    nib.save(nib.Nifti1Image(volume_data, np.eye(4)), volume_path)
    volume = _prepare(TemplateSpec("one_volume", "volume", "dwi", volume_path=volume_path), subject)
    endpoints = torch.tensor([[[1., 1, 1], [2., 2, 1]], [[2., 2, 1], [1., 1, 1]]])
    for second in (right, volume):
        actual = build_pair_connectomes(endpoints, left.labels, left.affine,
            second.labels, second.affine, first_node_count=1, second_node_count=1,
            radius=.1, weights=torch.tensor([1., 3.]), lengths=torch.tensor([10., 20.]),
            fa=torch.tensor([.25, .75]))
        assert actual["count"].tolist() == [[2]]
        assert actual["sift2_fbc"].tolist() == [[4.]]
        assert actual["mean_length"].tolist() == [[17.5]]
        assert actual["mean_fa"].tolist() == [[.625]]


@pytest.mark.parametrize("nodes", [False, True])
def test_unmapped_nonbackground_on_empty_hemisphere_still_fails(subject, nodes):
    spec = _spec(subject, nodes=nodes, other_values=(0, 55, 0))
    with pytest.raises(ValueError, match="absent from its node table"):
        preflight_template(spec, subject_dir=subject)
    with pytest.raises(ValueError, match="absent from its node table"):
        _prepare(spec, subject)


def test_wholly_background_surface_without_nodes_is_rejected(subject):
    paths = [_gifti(subject / f"label/{hemi}.empty.label.gii", (0, 0, 0), ((0, "background"),))
             for hemi in ("lh", "rh")]
    spec = TemplateSpec("empty", "surface", "native", left_path=paths[0], right_path=paths[1])
    with pytest.raises(ValueError, match="at least one"):
        preflight_template(spec, subject_dir=subject)
    with pytest.raises(ValueError, match="non-negative non-background ROIs"):
        _prepare(spec, subject)


def test_wholly_background_surface_with_a_declared_absent_node_keeps_it(subject):
    spec = _spec(subject, nodes=True)
    _gifti(spec.left_path, (0, 0, 0), ((0, "background"), (17, "one")))
    assert preflight_template(spec, subject_dir=subject)["node_count"] == 1
    prepared = _prepare(spec, subject)
    assert len(prepared.nodes) == 1 and not prepared.labels.any()


def test_empty_subset_accepts_only_declared_background_labels():
    nodes = (ConnectomeNode(1, 17, "L", "one"),)
    actual = _canonical_labels(np.array([0, -1, 0]), nodes, backgrounds=(0, -1),
                               device=torch.device("cpu"), hemisphere="R")
    assert actual.dtype == torch.int32 and actual.tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="absent from its node table"):
        _canonical_labels(np.array([0, -1, 0]), nodes, backgrounds=(0,),
                          device=torch.device("cpu"), hemisphere="R")


def test_nonempty_bilateral_canonical_path_is_exactly_unchanged():
    nodes = (ConnectomeNode(1, 17, "L", "left"), ConnectomeNode(2, 41, "L", "absent"),
             ConnectomeNode(3, 17, "R", "right"))
    values = np.array([17, -1, 0, 17])
    for side, expected in (("L", [1, 0, 0, 1]), ("R", [3, 0, 0, 3])):
        actual = _canonical_labels(values, nodes, backgrounds=(0, -1),
                                   device=torch.device("cpu"), hemisphere=side)
        assert actual.dtype == torch.int32 and actual.tolist() == expected
