"""Schema/identity controls; synthetic fixtures are not a benchmark."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("saved_reference", Path(__file__).with_name("collect_saved_reference.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_description():
    return {"device": "cuda:0", "motion_iterations": [1, 1, 1], "bold_reference_strategy": "robust",
            "bold_reference": {"status": "complete", "strategy": "robust", "input_frames": 180,
                               "selected_indices": list(range(20, 40)), "algorithm_dummy_scans": 0,
                               "skip_vols": 0, "discarded_input_frames": 0, "source_sha256": m.REFERENCE_SHA,
                               "dtype": "float32", "device": "cuda:0",
                               "parameters": {"n_volumes": 40, "zero_dummy_masked": 20, "nonnegative": True,
                                              "dummy_scans": None, "motion_correction": True, "stage_iterations": [1, 1, 1],
                                              "spatial_chunk_size": 262144},
                               "reference_motion": {"backend": "fnit.TorchMCFLIRT", "interpolation": "spline", "reference_selected_index": 0},
                               "outputs": {"bold_reference.nii.gz": {"sha256": "image", "bytes": 100}},
                               "timing_seconds": {"total": 2.0}}}


def test_full_saved_helper_accepts_and_rejects_wrong_identity():
    config = fixture_description()
    m.validate_description(config, "robust", "image", 100)
    with pytest.raises(ValueError, match="output identity"):
        m.validate_description(config, "robust", "wrong-image", 100)


@pytest.mark.parametrize("change", ["source", "motion", "iterations", "selection"])
def test_scientific_call_cannot_be_relabelled(change):
    config = fixture_description()
    if change == "source": config["bold_reference"]["source_sha256"] = "other"
    if change == "motion": config["bold_reference"]["reference_motion"]["backend"] = "AFNI"
    if change == "iterations": config["motion_iterations"] = [10, 10, 10]
    if change == "selection": config["bold_reference"]["selected_indices"] = [90]
    with pytest.raises(ValueError):
        m.validate_description(config, "robust", "image", 100)


def test_completed_report_raw_and_original_guards_required():
    raw = {"t1w": {"sha256": "t1"}, "bold": {"sha256": "bold"}}
    report = {"status": "scientific_complete", "case_id": "CON01", "variant": "robust", "frames": 180,
              "raw_hashes": {"t1w": "t1", "bold": "bold"}, "input_guards_equal": True, "source_guards_equal": True,
              "manifest_sha256": "config", "source_sha256": "source", "driver_sha256": m.CONTINUOUS_DRIVER_SHA}
    m.validate_report(report, "CON01", "robust", raw, "config", "source")
    for change in ({"status": "failed"}, {"source_guards_equal": None}, {"raw_hashes": {"t1w": "t1", "bold": "other"}}):
        with pytest.raises(ValueError):
            m.validate_report(dict(report, **change), "CON01", "robust", raw, "config", "source")


def test_real_timer_boundary_must_contain_helper_wall():
    description = fixture_description()["bold_reference"]
    metadata = {"FNIT": {"TimingSeconds": {"bold_reference": 2.1}}}
    assert m.validate_stage_seconds(metadata, description, "robust") == 2.1
    for invalid in (1.9, float("nan"), True, -1):
        metadata["FNIT"]["TimingSeconds"]["bold_reference"] = invalid
        with pytest.raises(ValueError):
            m.validate_stage_seconds(metadata, description, "robust")


def test_published_middle_requires_raw_frame_90_and_world_grid(tmp_path):
    data = np.arange(2 * 3 * 4 * 180, dtype=np.float32).reshape(2, 3, 4, 180)
    raw, middle = tmp_path / "raw.nii.gz", tmp_path / "middle.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), raw)
    nib.save(nib.Nifti1Image(data[..., 90], np.eye(4)), middle)
    m.middle_is_raw_frame(middle, raw)
    nib.save(nib.Nifti1Image(data[..., 91], np.eye(4)), middle)
    with pytest.raises(ValueError, match="raw middle frame"):
        m.middle_is_raw_frame(middle, raw)
    affine = np.eye(4); affine[0, 3] = 10
    nib.save(nib.Nifti1Image(data[..., 90], affine), middle)
    with pytest.raises(ValueError, match="physical grid"):
        m.middle_is_raw_frame(middle, raw)
