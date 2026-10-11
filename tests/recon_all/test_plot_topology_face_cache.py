"""绘图失败条件合同，不用小fixture代替真实MRI benchmark。"""
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from plot_recon_topology_face_detection_cache import run


def fixture_inputs(tmp_path, *, invalid=None):
    brain = tmp_path / "brain.mgz"
    nib.save(nib.MGHImage(np.zeros((3, 3, 3), np.uint8), np.eye(4)), brain)
    brain_sha = hashlib.sha256(brain.read_bytes()).hexdigest()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32)
    selected = np.array([0], np.int32)
    faces = np.array([[0, 1, 2]], np.int32)
    if invalid == "nan":
        vertices[0, 0] = np.nan
    if invalid == "negative":
        selected[0] = -1
    if invalid == "face_range":
        faces[0, 0] = 3
    paths = []
    for name in ("0-control", "1-candidate"):
        p = tmp_path / name / "candidate/A.final.private.npz"
        p.parent.mkdir(parents=True)
        np.savez(p, final_original_vertices=vertices, final_selected_vertices=selected, final_faces=faces)
        paths.append(p)
    report = tmp_path / "report.private.json"
    report.write_text(json.dumps(dict(status="passed", order=["control", "candidate", "candidate", "control"],
                                     records=[dict(input_sha256=dict(brain=brain_sha)) for _ in range(4)])))
    return brain, paths, report


@pytest.mark.parametrize("invalid", ["nan", "negative", "face_range"])
def test_invalid_geometry_rejected_before_png(tmp_path, invalid):
    brain, paths, report = fixture_inputs(tmp_path, invalid=invalid)
    out = tmp_path / "new.png"
    with pytest.raises(ValueError, match="finite corresponding"):
        run(brain=brain, baseline_candidate=paths[0], cached_candidate=paths[1],
            benchmark_report=report, output=out)
    assert not out.exists()


def test_wrong_brain_or_unbound_output_rejected(tmp_path):
    brain, paths, report = fixture_inputs(tmp_path)
    wrong = tmp_path / "other.mgz"
    nib.save(nib.MGHImage(np.ones((3, 3, 3), np.uint8), np.eye(4)), wrong)
    out = tmp_path / "new.png"
    with pytest.raises(ValueError, match="brain SHA"):
        run(brain=wrong, baseline_candidate=paths[0], cached_candidate=paths[1],
            benchmark_report=report, output=out)
    with pytest.raises(ValueError, match="actual baseline"):
        run(brain=brain, baseline_candidate=paths[1], cached_candidate=paths[1],
            benchmark_report=report, output=out)
    assert not out.exists()
