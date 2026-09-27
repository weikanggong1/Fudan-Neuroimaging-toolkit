import json

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.bids import locate_bids_inputs


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _nii(path, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    if len(shape) == 4:
        image.header["pixdim"][4] = 0.735
        image.header.set_xyzt_units(t="sec")
    nib.save(image, path)


def _dataset(root):
    _json(root / "dataset_description.json", {"Name": "Test", "BIDSVersion": "1.9.0"})
    _json(root / "task-rest_bold.json", {"TaskName": "Resting State", "RepetitionTime": 0.8})
    bold = root / "sub-01" / "ses-2" / "func" / "sub-01_ses-2_task-rest_run-1_bold.nii.gz"
    t1w = root / "sub-01" / "ses-2" / "anat" / "sub-01_ses-2_T1w.nii.gz"
    sbref = bold.with_name("sub-01_ses-2_task-rest_run-1_sbref.nii.gz")
    _nii(bold, (4, 5, 6, 8))
    _nii(t1w, (4, 5, 6))
    _nii(sbref, (4, 5, 6))
    _json(bold.with_name("sub-01_ses-2_task-rest_run-1_bold.json"), {
        "RepetitionTime": 0.735,
        "B0FieldSource": "gre1",
    })
    fmap = root / "sub-01" / "ses-2" / "fmap"
    phasediff = fmap / "sub-01_ses-2_acq-gre_phasediff.nii.gz"
    magnitude = fmap / "sub-01_ses-2_acq-gre_magnitude1.nii.gz"
    _nii(phasediff, (4, 5, 6))
    _nii(magnitude, (4, 5, 6))
    _json(phasediff.with_name("sub-01_ses-2_acq-gre_phasediff.json"), {
        "B0FieldIdentifier": "gre1",
        "EchoTime1": 0.004,
        "EchoTime2": 0.006,
    })
    # A derivative-like image in raw func and a separate derivatives tree are not inputs.
    _nii(bold.with_name("sub-01_ses-2_task-rest_desc-preproc_bold.nii.gz"), (4, 5, 6, 8))
    _nii(root / "derivatives" / "tool" / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz", (4, 5, 6, 8))
    return bold, t1w, sbref, phasediff, magnitude


def test_locates_one_raw_run_and_inherits_metadata(tmp_path):
    bold, t1w, sbref, phasediff, magnitude = _dataset(tmp_path)
    result = locate_bids_inputs(tmp_path, subject="sub-01", session="ses-2", run="1")
    assert (result.bold, result.t1w_images, result.sbref) == (bold, (t1w,), sbref)
    assert result.fieldmaps == (magnitude, phasediff)
    assert result.tr == 0.735
    assert result.bold_metadata["TaskName"] == "Resting State"
    assert len(result.bold_sidecars) == 2
    assert result.session == "2"


def test_entity_free_root_bold_json_is_inherited(tmp_path):
    bold, *_ = _dataset(tmp_path)
    (tmp_path / "task-rest_bold.json").unlink()
    root_sidecar = tmp_path / "bold.json"
    _json(root_sidecar, {"TaskName": "Resting State", "RepetitionTime": 0.8})

    result = locate_bids_inputs(tmp_path, subject="01", session="2", run="1")

    assert result.bold == bold
    assert result.bold_metadata["TaskName"] == "Resting State"
    assert result.tr == 0.735  # The run-specific JSON overrides the root TR.
    assert result.bold_sidecars == (
        root_sidecar, bold.with_name("sub-01_ses-2_task-rest_run-1_bold.json")
    )


def test_multiple_runs_require_a_selector(tmp_path):
    bold, *_ = _dataset(tmp_path)
    other = bold.with_name("sub-01_ses-2_task-rest_run-2_bold.nii.gz")
    _nii(other, (4, 5, 6, 8))
    _json(other.with_name("sub-01_ses-2_task-rest_run-2_bold.json"), {"RepetitionTime": 0.735})
    with pytest.raises(ValueError, match="exactly one raw BOLD run"):
        locate_bids_inputs(tmp_path, subject="01")
    assert locate_bids_inputs(tmp_path, subject="01", run="2").bold == other


def test_direction_and_reconstruction_select_exact_bold(tmp_path):
    bold, _, sbref, *_ = _dataset(tmp_path)
    bold.unlink()
    sbref.unlink()
    bold.with_name("sub-01_ses-2_task-rest_run-1_bold.json").unlink()
    _json(tmp_path / "task-rest_bold.json", {
        "TaskName": "Resting State", "RepetitionTime": 0.735,
    })
    func = bold.parent
    moco_ap = func / "sub-01_ses-2_task-rest_rec-moco_dir-AP_run-1_bold.nii.gz"
    moco_pa = func / "sub-01_ses-2_task-rest_rec-moco_dir-PA_run-1_bold.nii.gz"
    raw_ap = func / "sub-01_ses-2_task-rest_rec-raw_dir-AP_run-1_bold.nii.gz"
    for candidate in (moco_ap, moco_pa, raw_ap):
        _nii(candidate, (4, 5, 6, 8))

    with pytest.raises(ValueError, match="found 3"):
        locate_bids_inputs(tmp_path, subject="01", session="2", run="1")
    with pytest.raises(ValueError, match="found 2"):
        locate_bids_inputs(tmp_path, subject="01", run="1", direction="AP")
    with pytest.raises(ValueError, match="found 2"):
        locate_bids_inputs(tmp_path, subject="01", run="1", reconstruction="moco")
    assert locate_bids_inputs(tmp_path, subject="01", run="1",
                              reconstruction="rec-moco", direction="dir-AP").bold == moco_ap
    assert locate_bids_inputs(tmp_path, subject="01", run="1",
                              reconstruction="moco", direction="PA").bold == moco_pa
    with pytest.raises(ValueError, match="found 0"):
        locate_bids_inputs(tmp_path, subject="01", run="1", direction="LR")


def test_missing_tr_or_bad_shape_fails_before_processing(tmp_path):
    bold, *_ = _dataset(tmp_path)
    _json(bold.with_name("sub-01_ses-2_task-rest_run-1_bold.json"), {})
    (tmp_path / "task-rest_bold.json").unlink()
    with pytest.raises(ValueError, match="RepetitionTime"):
        locate_bids_inputs(tmp_path, subject="01")
    _json(tmp_path / "task-rest_bold.json", {"TaskName": "Rest", "RepetitionTime": 0.735})
    _nii(bold, (4, 5, 6))
    with pytest.raises(ValueError, match="Expected 4D"):
        locate_bids_inputs(tmp_path, subject="01")


def test_tr_must_agree_with_nifti_header(tmp_path):
    _dataset(tmp_path)
    _json(tmp_path / "sub-01" / "ses-2" / "func" / "sub-01_ses-2_task-rest_run-1_bold.json", {
        "RepetitionTime": 2.0,
    })
    with pytest.raises(ValueError, match="differs from NIfTI header"):
        locate_bids_inputs(tmp_path, subject="01")


def test_t1w_fallback_and_ambiguity(tmp_path):
    _, t1w, *_ = _dataset(tmp_path)
    t1w.unlink()
    subject_t1w = tmp_path / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    _nii(subject_t1w, (4, 5, 6))
    assert locate_bids_inputs(tmp_path, subject="01").t1w_images == (subject_t1w,)
    second = subject_t1w.with_name("sub-01_acq-second_T1w.nii.gz")
    _nii(second, (4, 5, 6))
    assert locate_bids_inputs(tmp_path, subject="01").t1w_images == tuple(sorted((second, subject_t1w)))


def test_derivative_root_is_rejected(tmp_path):
    _dataset(tmp_path)
    _json(tmp_path / "dataset_description.json", {
        "Name": "Preprocessed", "BIDSVersion": "1.9.0", "DatasetType": "derivative"
    })
    with pytest.raises(ValueError, match="raw BIDS dataset"):
        locate_bids_inputs(tmp_path, subject="01")
